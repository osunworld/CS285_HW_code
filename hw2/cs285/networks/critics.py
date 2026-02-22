import itertools  # (현재 파일에서는 직접 사용 안 함) 템플릿 import
from torch import nn  # 신경망 모듈
from torch.nn import functional as F  # 손실함수/함수형 연산
from torch import optim  # 옵티마이저

import numpy as np  # numpy 배열 처리
import torch  # 텐서 연산
from torch import distributions  # (현재 파일에서는 직접 사용 안 함) 템플릿 import

from cs285.infrastructure import pytorch_util as ptu  # MLP 빌더/디바이스/변환 유틸


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()  # nn.Module 초기화

        self.network = ptu.build_mlp(
            input_size=ob_dim,  # 입력: 관측 벡터
            output_size=1,  # 출력: 상태가치 V(s) 스칼라 1개
            n_layers=n_layers,  # hidden layer 개수
            size=layer_size,  # hidden layer 너비
        ).to(ptu.device)  # 계산 디바이스(CPU/GPU)로 이동

        self.optimizer = optim.Adam(
            self.network.parameters(),  # critic 네트워크 파라미터
            learning_rate,  # critic 학습률
        )  # Adam 옵티마이저 생성

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs).squeeze(-1)  # 네트워크 출력에서 마지막 차원 제거하여 (batch,) 형태로 반환

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)  # 관측 numpy -> torch 텐서
        q_values = ptu.from_numpy(q_values)  # 타깃 Q numpy -> torch 텐서

        # TODO: update the critic using the observations and q_values
        pred_values = self.forward(obs)  # 현재 critic이 예측한 V(s)
        loss = F.mse_loss(pred_values, q_values)  # MSE 손실 계산

        self.optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        self.optimizer.step()  # 파라미터 업데이트

        return {
            "Baseline Loss": ptu.to_numpy(loss),  # 로깅용 스칼라 값 반환
        }
