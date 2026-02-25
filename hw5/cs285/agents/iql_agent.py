from typing import Optional
import torch
from torch import nn
from cs285.agents.awac_agent import AWACAgent

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_value_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        # IQL은 AWAC와 달리 V(s) 네트워크를 별도로 둔다.
        # Q(s,a)와 V(s)를 분리 학습해 OOD action query를 피하는 것이 핵심.
        self.value_critic = make_value_critic(observation_shape)
        self.target_value_critic = make_value_critic(observation_shape)
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        # expectile 파라미터(보통 0.5~0.9):
        # 클수록 "상위 쪽" 값을 더 강조해 낙관적인 V를 만든다.
        self.expectile = expectile

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): Compute advantage with IQL
        # IQL advantage는 간단히
        # A(s,a_data) = Q(s,a_data) - V(s)
        # 형태를 쓴다. (AWAC처럼 V를 정책기대값으로 계산하지 않음)
        with torch.no_grad():
            qa_values = self.critic(observations)  # (batch, num_actions)
            actions = actions.long().view(-1, 1)  # (batch,) -> (batch, 1)
            q_values = qa_values.gather(1, actions).squeeze(1)  # (batch, 1) -> (batch,)
            vs = self.value_critic(observations).squeeze(-1)  # (batch, 1) -> (batch,)
            advantages = q_values - vs  # (batch,)
        return advantages
        return ...

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Update Q(s, a) to match targets (based on V)
        # IQL의 Q 업데이트 타깃:
        # y = r + gamma*(1-done)*V_target(s')
        # 즉, 다음 상태에서 정책으로 행동을 샘플링하지 않고
        # value critic을 통해 부트스트랩한다.
        with torch.no_grad():
            next_vs = self.target_value_critic(next_observations).squeeze(-1)
            target_values = rewards + self.discount * (1.0 - dones.float()) * next_vs

        qa_values = self.critic(observations)
        actions = actions.long().view(-1, 1)
        q_values = qa_values.gather(1, actions).squeeze(1)

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            # q_loss는 학습 손실 스칼라 로깅
            "q_loss": self.critic_loss(q_values, target_values).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Compute the expectile loss
        # expectile 회귀 손실:
        # diff = target_qs - vs
        # weight = expectile (diff>0) / (1-expectile) (diff<=0) 형태의 비대칭 가중
        # loss = mean(weight * diff^2)
        #
        # 직관:
        # - vs가 target_q보다 작을 때/클 때를 다르게 벌점 줘서
        #   평균(0.5 expectile)이 아닌 상단 통계량을 학습하게 만든다.
        diff = target_qs - vs
        weight = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1 - expectile))
        return (weight * diff * diff).mean()

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # TODO(student): Compute target values for V(s)
        # V 업데이트에서는 데이터셋 행동의 Q(s,a_data)를 타깃으로 사용.
        # 보통 target critic(또는 critic의 detach 값)을 이용해
        # V가 Q 분포의 expectile을 맞추도록 학습한다.

        # TODO(student): Update V(s) using the loss from the IQL paper
        # loss = iql_expectile_loss(expectile, vs, target_values)
        with torch.no_grad():
            qa_values = self.target_critic(observations)
            actions = actions.long().view(-1, 1)
            target_values = qa_values.gather(1, actions).squeeze(1)
        vs = self.value_critic(observations).squeeze(-1)
        loss = self.iql_expectile_loss(self.expectile, vs, target_values)

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            # vs_adv는 V와 Q target의 차이를 모니터링하기 위한 진단 지표
            "vs_adv": (vs - target_values).mean().item(),
            "vs": vs.mean().item(),
            "target_values": target_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """
        # IQL critic 단계는 Q와 V를 모두 업데이트해야 complete
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        # 1) critic(Q,V) 업데이트
        metrics = self.update_critic(observations, actions, rewards, next_observations, dones)
        # 2) actor는 AWAC 형태의 advantage-weighted BC로 업데이트
        metrics["actor_loss"] = self.update_actor(observations, actions)

        # 3) 주기적으로 target 네트워크 동기화
        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()
        
        return metrics

    def update_target_value_critic(self):
        # hard update: V_target <- V
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
