import itertools  # 여러 파라미터 iterable을 하나로 묶을 때 사용
from torch import nn  # 신경망 모듈 기본 클래스/레이어
from torch.nn import functional as F  # 손실함수/함수형 연산 모음
from torch import optim  # 옵티마이저(Adam 등)

import numpy as np  # numpy 배열 처리
import torch  # 텐서 연산
from torch import distributions  # 행동 분포(Categorical/Normal) 생성

from cs285.infrastructure import pytorch_util as ptu  # numpy<->torch, MLP 빌더, device 유틸


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()  # nn.Module 초기화

        if discrete:  # 이산 행동공간인 경우
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,  # 입력: 관측 벡터
                output_size=ac_dim,  # 출력: 각 행동의 로짓(logit)
                n_layers=n_layers,  # hidden layer 개수
                size=layer_size,  # hidden layer 너비
            ).to(ptu.device)  # 계산 디바이스(CPU/GPU)로 이동
            parameters = self.logits_net.parameters()  # 학습 대상 파라미터
        else:  # 연속 행동공간인 경우
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,  # 입력: 관측 벡터
                output_size=ac_dim,  # 출력: 행동 평균(mean)
                n_layers=n_layers,  # hidden layer 개수
                size=layer_size,  # hidden layer 너비
            ).to(ptu.device)  # 계산 디바이스로 이동
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)  # 행동 차원별 log std 초기값
            )  # 학습 가능한 파라미터로 등록
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())  # mean_net + logstd 함께 최적화

        self.optimizer = optim.Adam(
            parameters,  # 위에서 모은 학습 대상
            learning_rate,  # 학습률
        )  # Adam 옵티마이저 생성

        self.discrete = discrete  # 행동공간 타입 플래그 저장

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        
        obs = ptu.from_numpy(obs)  # numpy 관측 -> torch 텐서
        dist = self.forward(obs)  # forward pass로 행동 분포 생성
        action = dist.sample()  # 분포에서 샘플링하여 행동 생성
        action = ptu.to_numpy(action)  # torch 텐서 -> numpy 배열

        return action  # 단일 관측에 대한 단일 행동 반환

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:  # 이산 행동이면 Categorical 분포를 보통 사용
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)  # 관측에서 행동 로짓 계산
            dist = distributions.Categorical(logits=logits)  # Categorical 분포 생성
        else:  # 연속 행동이면 Normal 분포를 보통 사용
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)  # 관측에서 행동 평균 계산
            std = torch.exp(self.logstd)  # logstd -> std로 변환
            dist = distributions.Normal(mean, std)  # Normal 분포 생성
        return dist  # 분포 객체 반환

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError  # 알고리즘별 업데이트 규칙은 하위 클래스에서 구현


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)  # numpy 관측 -> torch 텐서
        actions = ptu.from_numpy(actions)  # numpy 행동 -> torch 텐서
        advantages = ptu.from_numpy(advantages)  # numpy advantage -> torch 텐서
        
        # TODO: implement the policy gradient actor update.
        dist = self.forward(obs)  # forward pass로 행동 분포 생성
        log_probs = dist.log_prob(actions)  # 각 행동에 대한 log probability 계산
        if not self.discrete:
            log_probs = log_probs.sum(dim=-1)
        loss = -(log_probs * advantages).mean()
        self.optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        self.optimizer.step()  # 파라미터 업데이트

        return {
            "Actor Loss": ptu.to_numpy(loss),  # 로깅을 위해 다시 numpy/스칼라로 변환
        }
