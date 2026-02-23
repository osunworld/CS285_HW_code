from typing import Tuple, Optional

import gym

import numpy as np
import torch
import torch.nn as nn

from cs285.networks.mlp_policy import MLPPolicy
from cs285.networks.state_action_value_critic import StateActionCritic
import cs285.infrastructure.pytorch_util as ptu

from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


def sac_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    total_steps: int = 300000,
    random_steps: int = 5000,
    training_starts: int = 10000,
    batch_size: int = 128,
    replay_buffer_capacity: int = 1000000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    use_soft_target_update: bool = False,
    target_update_period: Optional[int] = None,
    soft_target_update_rate: Optional[float] = None,
    # Actor-critic configuration
    actor_gradient_type="reinforce",  # One of "reinforce" or "reparametrize"
    num_actor_samples: int = 1,
    num_critic_updates: int = 1,
    # Settings for multiple critics
    num_critic_networks: int = 1,
    target_critic_backup_type: str = "mean",  # One of "doubleq", "min", or "mean"
    # Soft actor-critic
    backup_entropy: bool = True,
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,
    actor_fixed_std: Optional[float] = None,
    use_tanh: bool = True,
):
    # ------------------------------------------------------------------
    # 이 함수는 "실험 설정(dict)"을 만들어 run_hw3_sac.py로 넘겨준다.
    # 핵심은 아래 make_* 팩토리 함수들:
    # - make_actor / make_critic: 네트워크 구조
    # - make_*_optimizer: 최적화기
    # - make_lr_schedule: 학습률 스케줄
    # - make_env: 환경 wrapper 체인
    # ------------------------------------------------------------------
    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        # SAC critic은 Q(s,a)를 직접 회귀하므로 입력 차원은 ob_dim + ac_dim
        return StateActionCritic(
            ob_dim=np.prod(observation_shape),
            ac_dim=action_dim,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        assert len(observation_shape) == 1
        if actor_fixed_std is not None:
            # 표준편차를 상수로 고정: 학습 안정성 디버깅 시 유용
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=False,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=False,
                fixed_std=actor_fixed_std,
            )
        else:
            # 상태의존 표준편차: 일반적으로 표현력이 더 높음
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=False,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=True,
            )

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        # actor/critic을 분리 학습할 수 있게 LR를 따로 둔다.
        return torch.optim.Adam(params, lr=actor_learning_rate)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        # 기본 과제에서는 고정 학습률 사용
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        # wrapper 순서:
        # 1) gym.make
        # 2) RescaleAction: 환경 행동 범위를 [-1,1]로 표준화
        # 3) ClipAction: 행동 클리핑으로 안정성 보강
        # 4) RecordEpisodeStatistics: episode return/length 자동 기록
        return RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    gym.make(
                        env_name, render_mode="single_rgb_array" if render else None
                    ),
                    -1,
                    1,
                )
            )
        )

    # 로그 디렉토리 이름에 주요 하이퍼파라미터를 인코딩
    # (실험 결과 비교 시 매우 중요)
    log_string = "{}_{}_{}_s{}_l{}_alr{}_clr{}_b{}_d{}".format(
        exp_name or "offpolicy_ac",
        env_name,
        actor_gradient_type,
        hidden_size,
        num_layers,
        actor_learning_rate,
        critic_learning_rate,
        batch_size,
        discount,
    )

    if use_entropy_bonus:
        log_string += f"_t{temperature}"

    if use_soft_target_update:
        log_string += f"_stu{soft_target_update_rate}"
    else:
        log_string += f"_htu{target_update_period}"

    if target_critic_backup_type != "mean":
        log_string += f"_{target_critic_backup_type}"

    return {
        "agent_kwargs": {
            # TODO 구현 시 주의:
            # SoftActorCritic.__init__ 시그니처와 key 이름이 일치해야 한다.
            "make_critic": make_critic,
            "make_critic_optimizer": make_actor_optimizer,
            "make_critic_schedule": make_lr_schedule,
            "make_actor": make_actor,
            "make_actor_optimizer": make_critic_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "num_critic_updates": num_critic_updates,
            "discount": discount,
            "actor_gradient_type": actor_gradient_type,
            "num_actor_samples": num_actor_samples,
            "num_critic_updates": num_critic_updates,
            "num_critic_networks": num_critic_networks,
            "target_critic_backup_type": target_critic_backup_type,
            "use_entropy_bonus": use_entropy_bonus,
            "backup_entropy": backup_entropy,
            "temperature": temperature,
            "target_update_period": target_update_period
            if not use_soft_target_update
            else None,
            "soft_target_update_rate": soft_target_update_rate
            if use_soft_target_update
            else None,
        },
        # replay buffer 용량이 작으면 off-policy 학습 다양성이 급격히 줄어든다.
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
    }
