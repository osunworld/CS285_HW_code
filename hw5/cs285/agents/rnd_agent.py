import torch
from torch import nn
import numpy as np

from typing import Callable, List, Tuple

from cs285.agents.dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu

def init_network(model):
    # target RND 네트워크는 "고정된(random)" 함수여야 하므로,
    # 일반적인 초기화 대신 강한 랜덤 초기화를 적용한다.
    # 이렇게 하면 초기에 상태별 예측오차가 충분히 다양하게 생긴다.
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()

class RNDAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        make_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        make_rnd_network_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        make_target_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        rnd_weight: float,
        **kwargs
    ):
        # RNDAgent는 DQN 기반 탐욕/탐험 행동선택 틀을 재사용한다.
        # 즉, 기본 Q-learning 파이프라인(critic/target/replay/update)은
        # 부모 클래스 DQNAgent가 담당한다.
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        # RND 보너스를 외부 보상(extrinsic reward)에 얼마나 섞을지 결정.
        # 수정된 보상: r_total = r_env + rnd_weight * r_rnd
        self.rnd_weight = rnd_weight

        # 학습 대상 predictor 네트워크 f_hat(s)
        self.rnd_net = make_rnd_network(observation_shape)
        # 고정 target 네트워크 f*(s)
        self.rnd_target_net = make_target_rnd_network(observation_shape)

        # target 네트워크는 랜덤 함수 역할을 하므로 초기화 후 고정한다.
        self.rnd_target_net.apply(init_network)

        # Freeze target network
        for p in self.rnd_target_net.parameters():
            p.requires_grad_(False)

        self.rnd_optimizer = make_rnd_network_optimizer(
            self.rnd_net.parameters()
        )

    def update_rnd(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update the RND network using the observations.
        """
        # TODO(student): update the RND network
        # 구현 의도:
        # 1) target = rnd_target_net(obs) (no grad)
        # 2) pred = rnd_net(obs)
        # 3) loss = 평균 예측오차 (보통 MSE 또는 L2 norm)
        # 여기서 학습되는 것은 rnd_net 하나뿐이며,
        # rnd_target_net은 고정된 "비교 기준"으로만 사용된다.
        with torch.no_grad():
            target = self.rnd_target_net(obs)
        pred = self.rnd_net(obs)
        loss = torch.mean((pred - target) ** 2)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            # TODO(student): Compute RND bonus for batch and modify rewards
            # 핵심:
            # - next_observations를 입력으로 target/predictor 출력을 비교해
            #   상태별 novelty 점수(rnd_error)를 만든다.
            # - shape은 rewards와 동일한 (batch,)이어야 한다.
            # - 보너스를 rewards에 더해 DQN 업데이트에 전달한다.
            #   rewards <- rewards + rnd_weight * rnd_error
            with torch.no_grad():
                target = self.rnd_target_net(next_observations)
                pred = self.rnd_net(next_observations)
                rnd_error = torch.norm(pred - target, dim=-1)
            assert rnd_error.shape == rewards.shape
            rewards = rewards + self.rnd_weight * rnd_error

        # 보너스가 반영된 보상으로 기존 DQN 업데이트 수행
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the RND network.
        # 관측 분포에 맞춰 predictor를 지속적으로 맞추면,
        # 자주 방문한 상태는 에러가 줄고, 새로운 상태는 에러가 크게 남는다.
        rnd_loss = self.update_rnd(observations)
        metrics["rnd_loss"] = rnd_loss

        return metrics

    def num_aux_plots(self) -> int:
        return 1
    
    def plot_aux(
        self,
        axes: List,
    ) -> dict:
        """
        Plot the RND prediction error for the observations.
        """
        import matplotlib.pyplot as plt
        assert len(axes) == 1
        ax: plt.Axes = axes[0]

        with torch.no_grad():
            # Assume a state space of [0, 1] x [0, 1]
            x = torch.linspace(0, 1, 100)
            y = torch.linspace(0, 1, 100)
            xx, yy = torch.meshgrid(x, y)

            inputs = ptu.from_numpy(np.stack([xx.flatten(), yy.flatten()], axis=1))
            targets = self.rnd_target_net(inputs)
            predictions = self.rnd_net(inputs)

            # 상태공간 전역에서 오차 히트맵을 그려
            # 어디를 "새롭다(novel)"고 보는지 시각화한다.
            errors = torch.norm(predictions - targets, dim=-1)
            errors = torch.reshape(errors, xx.shape)

            # Log scale, aligned with normal axes
            from matplotlib import cm
            ax.imshow(ptu.to_numpy(errors).T, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
            plt.colorbar(ax.images[0], ax=ax)
