from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        # backup 방식/actor gradient 방식 유효성 검사
        # (설정 오타가 나면 조기 실패시켜 디버깅 시간을 줄인다.)
        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        # actor: 연속 행동 분포 π(a|s)를 출력하는 정책 네트워크
        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        # critic ensemble: Q(s,a) 네트워크를 여러 개 둘 수 있다.
        # - 1개면 기본 Q-learning
        # - 2개 이상이면 double/min/reduced bias 실험 가능
        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        # target critics: bootstrapping 타깃 계산용 "느리게 변하는" 네트워크
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        # 초기에는 critic과 target critic을 동일하게 맞춘다.
        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            # 단일 관측 -> 배치 차원 1개를 갖는 텐서로 변환
            observation = ptu.from_numpy(observation)[None]

            # actor는 행동 "값"이 아니라 행동 "분포"를 반환한다.
            # SAC는 이 분포에서 샘플링한 행동으로 상호작용한다.
            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        # 출력 shape: (num_critics, batch_size)
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        # target critic도 동일하게 앙상블 shape으로 반환
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        # 핵심 아이디어:
        # - doubleq: 서로의 값을 타깃으로 쓰도록 critic 축을 교환(2개 기준)
        # - min: 과대추정을 줄이기 위해 critic들 중 최소값 사용
        # - mean: critic 평균값 사용
        # 이 함수의 최종 출력은 critic 개수 축을 유지한 (num_critics, batch)여야 한다.
        if self.target_critic_backup_type == "doubleq":
            assert self.num_critic_networks == 2, "Double-Q backup only works with 2 critics"
            next_qs = next_qs.flip(0)  # 두 critic의 축을 교환
        elif self.target_critic_backup_type == "min":
            next_qs = torch.min(next_qs, dim=0).values  # critic들 중 최소값 선택
        elif self.target_critic_backup_type == "mean":
            next_qs = torch.mean(next_qs, dim=0)  # critic들의 평균값 사용
        else:
            # Default, we don't need to do anything.
            pass


        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        # reward/done은 보통 (batch,) 벡터
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            # Sample from the actor
            # 다음 상태 s_{t+1}에서 다음 행동 a_{t+1}를 정책에서 샘플링
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()  # shape (num_actor_samples, batch_size, action_dim)

            # Compute the next Q-values for the sampled actions
            # Q_target(s_{t+1}, a_{t+1})를 앙상블 critic들로 계산
            next_qs = self.target_critic(next_obs, next_action)  # shape (num_critic_networks, num_actor_samples, batch_size)
            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                # SAC에서는 타깃에 alpha * H(pi(.|s_{t+1}))를 더해 탐험성을 보존
                next_action_entropy = -next_action_distribution.log_prob(next_action) # shape (num_actor_samples, batch_size)
                next_qs += temperature * next_action_entropy[None]  # actor 샘플 축 평균

            # Compute the target Q-value
            # Bellman 타깃:
            # y = r + gamma * (1 - done) * next_q
            target_values: torch.Tensor = reward[None] + self.discount * (1 - done[None]) * next_qs
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Update the critic
        # Predict Q-values
        # 현재 시점 (s_t, a_t)에 대한 Q 예측
        q_values = self.critic(obs, action)  # shape (num_critic_networks, batch_size)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        # critic 손실 = 예측 Q와 bootstrapped target 간 MSE
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        # 팁:
        # - 연속 분포는 analytic entropy(action_distribution.entropy())를 지원할 수 있다.
        # - tanh-squash를 쓴다면 analytic entropy가 정확하지 않아 샘플 기반 근사를 쓰기도 한다.
        return -action_distribution.log_prob(action_distribution.rsample())  # shape (batch_size,)

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        K = self.num_actor_samples

        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            action = action_distribution.sample((K,))  # (K, B, A)
            assert action.shape == (K, batch_size, self.action_dim), action.shape

            # obs를 K번 복제해서 action과 같은 leading shape로 맞춘 뒤 flatten
            obs_tiled = (
                obs.unsqueeze(0)
                .expand(K, batch_size, *obs.shape[1:])
                .reshape(K * batch_size, *obs.shape[1:])
            )
            action_flat = action.reshape(K * batch_size, self.action_dim)

            q_values = self.critic(obs_tiled, action_flat)  # (N, K*B)
            q_values = q_values.reshape(self.num_critic_networks, K, batch_size)
            assert q_values.shape == (self.num_critic_networks, K, batch_size), q_values.shape

            q_values = torch.mean(q_values, axis=0)  # (K, B)
            advantage = q_values

        log_probs = action_distribution.log_prob(action)  # (K, B)
        loss = -torch.mean(log_probs * advantage)

        return loss, torch.mean(self.entropy(action_distribution))


    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        # 재매개변수화 방식은 정책 샘플로 직접 미분을 흘릴 수 있다.
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        # reparametrize라면 보통 rsample()을 사용해 pathwise gradient를 전달
        action = action_distribution.rsample()  # shape (batch_size, action_dim)

        # TODO(student): Compute Q-values for the sampled state-action pair
        # 보통 앙상블 평균 Q를 사용
        q_values = self.critic(obs, action)  # shape (num_critic_networks, batch_size)

        # TODO(student): Compute the actor loss
        # 직관: actor가 Q가 큰 행동을 더 자주 내도록 -Q를 최소화
        loss = -q_values.mean()  # shape ()
        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            # maximize (Q + alpha * entropy) <=> minimize -(Q + alpha * entropy)
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        # hard update: tau=1이면 target <- critic 완전 복사
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        # Polyak averaging:
        # target = (1-tau)*target + tau*online
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        # num_critic_updates > 1이면 actor 1번 업데이트 전에 critic을 여러 번 학습해
        # Q 추정을 더 안정화한다.
        for _ in range(self.num_critic_updates):
            critic_info = self.update_critic(
                obs=observations,
                action=actions,
                reward=rewards,
                next_obs=next_observations,
                done=dones,
            )
            critic_infos.append(critic_info)
        # TODO(student): Update the actor
        # actor는 최신 critic 기준으로 E[Q]를 키우는 방향으로 1회 업데이트
        actor_info = self.update_actor(observations)

        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        # 구현 의도:
        # - hard update: step % target_update_period == 0일 때만 전체 복사
        # - soft update: 매 step마다 tau=soft_target_update_rate로 부분 반영
        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif step % self.target_update_period == 0:
            self.update_target_critic()
        # Average the critic info over all of the steps
        # critic을 여러 번 업데이트했으므로 로그는 평균으로 요약
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
