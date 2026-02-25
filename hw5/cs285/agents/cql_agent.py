from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        # CQL 정규화 항의 계수(alpha):
        # - 클수록 보수적으로(Q 낮게) 학습됨
        # - 너무 크면 과도한 과소추정으로 성능 저하 가능
        self.cql_alpha = cql_alpha
        # logsumexp 계산 시 스케일 조절(선택적으로 사용 가능)
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        # CQL 핵심 아이디어:
        # - 모든 행동에 대한 soft max(logsumexp Q)를 낮추고
        # - 데이터셋에서 실제 관측된 행동 Q(s,a_data)는 상대적으로 높인다.
        #
        # 전형적인 regularizer:
        # cql_reg = E_s[ logsumexp(Q(s,·)) - Q(s,a_data) ]
        # 최종 손실:
        # loss = td_loss + cql_alpha * cql_reg
        #
        # 참고:
        # - qa_values: shape (batch, num_actions)
        # - q_values:  shape (batch,)
        # - 수치안정성 때문에 torch.logsumexp 사용 권장
        qa_values = variables['qa_values']  # (batch, num_actions)
        q_values = variables['q_values']  # (batch,)
        cql_logsumexp = torch.logsumexp(qa_values / self.cql_temperature, dim=1) * self.cql_temperature  # (batch,)
        cql_reg = cql_logsumexp - q_values  # (batch,)
        loss = loss + self.cql_alpha * cql_reg.mean()  # CQL
        metrics['cql_reg'] = cql_reg.mean().item()  # CQL 정규화 항의 평균값도 로깅

        return loss, metrics, variables
