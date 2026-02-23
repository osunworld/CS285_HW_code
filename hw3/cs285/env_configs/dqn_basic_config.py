from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
import cs285.infrastructure.pytorch_util as ptu

def basic_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 300000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 128,
    **kwargs
):
    # 이 함수는 팩토리 함수들과 하이퍼파라미터를 담은 딕셔너리를 반환한다.
    # `run_hw3_dqn.py`는 이 딕셔너리를 받아 env/agent/schedule을 구성한다.

    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        # 기본 제어 환경은 보통 저차원 벡터 관측을 제공한다.
        # 따라서 평탄화된 관측을 받아 각 행동의 Q값을 출력하는 MLP를 사용한다.
        # 출력 shape은 [batch, num_actions]이며, 각 이산 행동당 Q값 1개다.
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        # 이 과제의 DQN 기본 옵티마이저는 Adam이다.
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        # 다른 설정이 없으면 학습률을 상수로 유지한다.
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # 입실론-그리디 탐험 스케줄.
    # 처음엔 완전 랜덤(eps=1)으로 시작해서 전체 스텝의 10% 지점까지 0.02로 감소.
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (total_steps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )

    def make_env(render: bool = False):
        # RecordEpisodeStatistics는 에피소드 return/length를 `info`에 기록한다.
        # 이 값은 로거에서 학습 곡선을 그릴 때 사용된다.
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None))

    # 주요 하이퍼파라미터를 로그 이름에 넣어 실험 비교를 쉽게 한다.
    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "dqn",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    if use_double_q:
        # Double-DQN 사용 여부를 로그 이름에 표시해 실험을 구분한다.
        log_string += "_doubleq"

    # 반환 딕셔너리의 필드는 학습 스크립트가 기대하는 공통 인터페이스다.
    # `agent_kwargs`는 DQNAgent(...) 생성자로 그대로 전달된다.
    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "exploration_schedule": exploration_schedule,
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
