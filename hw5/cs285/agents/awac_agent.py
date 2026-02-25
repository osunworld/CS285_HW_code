from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        # AWAC는 critic 외에 확률정책(actor)을 따로 학습한다.
        # actor는 "데이터셋 행동을 가중 모방"하는 방식으로 업데이트된다.
        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        # advantage 가중치 exp(A / temperature)에서 분모 역할
        # temperature가 작을수록 높은 A 행동에 더 집중한다.
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            # 1) 다음 상태에서 actor 분포 pi(a|s')를 구하고
            # 2) target_critic의 Q(s',a')를 분포로 평균내어 E_{a'~pi}[Q(s',a')] 계산
            # 3) TD target = r + gamma*(1-done)*E[Q_next]
            next_action_dist = self.actor(next_observations)
            next_qa_values = self.target_critic(next_observations)

            # Use the actor to compute a critic backup

            next_qs = torch.sum(next_action_dist.probs * next_qa_values, dim=-1)

            # TODO(student): Compute the TD target
            target_values = rewards.float() + self.discount * (1.0 - dones.float()) * next_qs

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        # 현재 상태-행동의 Q(s,a_data)를 critic에서 꺼내
        # TD target과 MSE로 회귀한다.
        # (AWAC도 critic 업데이트 자체는 TD 학습 기반)
        qa_values = self.critic(observations)
        actions = actions.long().view(-1, 1)
        q_values = qa_values.gather(1, actions).squeeze(1)
        assert q_values.shape == target_values.shape

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

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        # advantage 정의:
        # A(s,a_data) = Q(s,a_data) - V(s)
        # 여기서 V(s)는 actor 분포에 대한 기대값 E_{a~pi}[Q(s,a)]로 근사한다.
        #
        # 구현 팁:
        # - qa_values: critic(observations) -> (batch, num_actions)
        # - q_values: 데이터 행동 인덱스로 gather -> (batch,)
        # - values: actor 분포(prob)와 qa_values의 가중합 -> (batch,)
        if action_dist is None:
            action_dist = self.actor(observations)

        qa_values = self.critic(observations)
        actions = actions.long().view(-1, 1)
        q_values = qa_values.gather(1, actions).squeeze(1)
        values = torch.sum(action_dist.probs * qa_values, dim=-1)

        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        # AWAC actor loss (가중 행동모방):
        # L = - E[ log pi(a_data|s) * exp(A(s,a_data)/temperature) ]
        #
        # 실무적으로는 exp 폭주를 막기 위해
        # - advantage clip
        # - weight clip/max cap
        # 등을 적용하기도 한다.
        action_dist = self.actor(observations)
        actions = actions.long().view(-1)

        advantages = self.compute_advantage(observations, actions, action_dist)
        weights = torch.exp(advantages / self.temperature).detach()
        weights = torch.clamp(weights, max=100.0)

        log_probs = action_dist.log_prob(actions)
        loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        # 1) critic(DQN 기반 TD) 업데이트
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        # 2) 같은 배치로 actor(가중 BC) 업데이트
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
