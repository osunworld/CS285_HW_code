from typing import Optional, Tuple

import gym
from gym.wrappers.frame_stack import FrameStack

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
from cs285.infrastructure.atari_wrappers import wrap_deepmind
import cs285.infrastructure.pytorch_util as ptu


class PreprocessAtari(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 기대 입력은 스택된 흑백 프레임 (4, 84, 84) 또는 배치 형태다.
        # 래퍼/설정이 맞지 않으면 assert로 초기에 바로 실패시킨다.
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        # CNN 학습 안정성을 위해 uint8 픽셀 [0,255]을 float [0,1]로 정규화한다.
        return x / 255.0


def atari_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    learning_rate: float = 1e-4,
    adam_eps: float = 1e-4,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 2000,
    clip_grad_norm: Optional[float] = 10.0,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 32,
    **kwargs,
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        # Atari 래퍼를 거치면 관측은 4장 스택된 84x84 프레임이어야 한다.
        # 환경/래퍼 설정 오류를 즉시 잡기 위해 assert를 유지한다.
        assert observation_shape == (
            4,
            84,
            84,
        ), f"Observation shape: {observation_shape}"

        # 전형적인 DQN CNN 구조:
        #   conv(8x8,s4) -> conv(4x4,s2) -> conv(3x3,s1) -> FC(512) -> 행동별 Q값
        # Atari에서 표준적으로 쓰이는 구조이며 공간/시간 패턴을 잘 포착한다.
        return nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, num_actions),
        ).to(ptu.device)

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        # Atari DQN에서는 비기본 epsilon을 둔 Adam 설정이 자주 사용된다.
        return torch.optim.Adam(params, lr=learning_rate, eps=adam_eps)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        # 구간별 학습률 스케줄:
        #   초반에는 유지하고, 전체 학습 중간쯤에서 0.5배로 감소.
        # PiecewiseSchedule에서 lambda(step)를 받아 LambdaLR로 적용한다.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            PiecewiseSchedule(
                [
                    (0, 1),
                    (20000, 1),
                    (total_steps / 2, 5e-1),
                ],
                outside_value=5e-1,
            ).value,
        )

    # Atari용 입실론-그리디 탐험 스케줄:
    # 초반 랜덤 탐험을 크게 주고, 이후 0.01까지 천천히 감소시킨다.
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (20000, 1),
            (total_steps / 2, 0.01),
        ],
        outside_value=0.01,
    )

    def make_env(render: bool = False):
        # `wrap_deepmind`는 Atari DQN에 중요한 전처리를 적용한다:
        # 프레임 스킵, 흑백/리사이즈, 프레임 스택, 에피소드 통계 기록.
        return wrap_deepmind(
            gym.make(env_name, render_mode="rgb_array" if render else None)
        )

    # 주요 설정을 로그 이름에 포함해 실험 폴더만 봐도 설정을 알 수 있게 한다.
    log_string = "{}_{}_d{}_tu{}_lr{}".format(
        exp_name or "dqn",
        env_name,
        discount,
        target_update_period,
        learning_rate,
    )

    if use_double_q:
        # Double-DQN 사용 실험임을 폴더 이름에 표시한다.
        log_string += "_doubleq"

    if clip_grad_norm is not None:
        # 그래디언트 클리핑 값은 Atari 안정성에 큰 영향을 줄 수 있다.
        log_string += f"_clip{clip_grad_norm}"

    # 공통 DQN 학습 스크립트가 사용하는 설정 묶음을 반환한다.
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
        "log_name": log_string,
        "exploration_schedule": exploration_schedule,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
