from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        # 온라인 Q 네트워크: 실제 gradient descent로 업데이트되는 네트워크
        self.critic = make_critic(observation_shape, num_actions)
        # 타깃 Q 네트워크: Bellman target 계산 안정화를 위해 사용
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        # 시작 시점에 두 네트워크 파라미터를 맞춰둔다.
        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Used for evaluation.
        """
        # (obs_dim,) -> (1, obs_dim): 배치 차원 추가
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # epsilon-greedy 규칙:
        # - 확률 epsilon: 랜덤 행동(탐험)
        # - 확률 1-epsilon: argmax_a Q(s,a) (활용)
        # 구현 시 action은 배치 차원을 유지한 텐서 형태가 이후 처리에 편하다.
        if np.random.rand() < epsilon:
            action = torch.randint(low=0, high=self.num_actions, size=(1,),device=observation.device)
        else:
            with torch.no_grad():
                qa_values = self.critic(observation)
                action = qa_values.argmax(dim=-1)  # (1,)

        # 최종 반환은 단일 정수 action id
        return ptu.to_numpy(action).squeeze(0).item()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """

        # TODO(student): paste in your code from HW3, and make sure the return values exist
        # 구현 핵심 흐름:
        # 1) qa_values = critic(obs): 모든 행동의 Q(s,·)
        # 2) q_values = qa_values.gather(...): 실제 action의 Q(s,a)
        # 3) target 계산:
        #    - 기본 DQN: a' = argmax_a Q_target(s',a)
        #    - Double DQN: a' 선택은 online(critic), 평가는 target_critic
        # 4) target_values = r + gamma*(1-done)*Q_target(s',a')
        # 5) loss = MSE(q_values, target_values)
        #
        # 주의:
        # - target 계산은 반드시 no_grad 블록에서 수행
        # - done은 부트스트랩 차단 마스크로 사용
        # - shape 체크(batch,)를 통일해두면 디버깅이 쉬움
        batch_size = reward.shape[0]
        action = action.long().view(-1, 1)  # (batch,) -> (batch, 1) 형태로 변환해 gather에 사용하기 편하게 한다.
        reward = reward.float().view(-1)  # (batch,) 형태로 통일
        done = done.float().view(-1)  # (batch,) 형태로 통일
        qa_values = self.critic(obs)  # (batch, num_actions)
        q_values = qa_values.gather(1,action).squeeze(1)  # (batch, 1) -> (batch,) 실제 행동의 Q값

        with torch.no_grad():
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # Double DQN: 행동선택(online), 행동평가(target) 분리
                
                next_action = self.critic(next_obs).argmax(dim=-1, keepdim=True)
            else:
                # Vanilla DQN: 행동선택+평가 둘 다 target 사용
                next_action = self.target_critic(next_obs).argmax(dim=-1, keepdim=True)

            next_q_values = self.target_critic(next_obs).gather(1, next_action).squeeze(1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + (1 - done) * self.discount * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape
        loss = self.critic_loss(q_values, target_values)


        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        # 손실/메트릭 계산은 compute_critic_loss에서 담당
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        # gradient clipping은 불안정한 업데이트(gradient 폭주) 완화용
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        # hard update: target <- online
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): paste in your code from HW3
        # 구현 의도:
        # - 매 step마다 critic 업데이트
        # - target_update_period마다 target_critic 동기화(hard update)
        # - 최종적으로 critic_stats를 반환해 로깅에서 사용
        #
        # 예시 구조:
        # critic_stats = self.update_critic(...)
        # if step % self.target_update_period == 0:
        #     self.update_target_critic()
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        return critic_stats
