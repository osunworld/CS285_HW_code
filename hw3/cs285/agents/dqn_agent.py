from typing import Sequence, Callable, Tuple, Optional  # 타입 힌트용 유틸 임포트

import torch  # 텐서 계산/자동미분 라이브러리
from torch import nn  # 신경망 모듈 기본 클래스

import numpy as np  # 넘파이 배열 연산

import cs285.infrastructure.pytorch_util as ptu  # numpy<->torch 변환/디바이스 유틸


class DQNAgent(nn.Module):  # DQN 에이전트(critic + target critic) 클래스
    def __init__(
        self,
        observation_shape: Sequence[int],  # 관측(shape) 정보
        num_actions: int,  # 이산 행동 개수
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],  # Q네트워크 생성 함수
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],  # 옵티마이저 팩토리
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],  # 학습률 스케줄러 팩토리
        discount: float,  # 감가율 gamma
        target_update_period: int,  # 타깃 네트워크 하드 업데이트 주기
        use_double_q: bool = False,  # Double-DQN 사용 여부
        clip_grad_norm: Optional[float] = None,  # 그래디언트 클리핑 임계값
    ):
        super().__init__()  # nn.Module 초기화

        # 온라인 Q 네트워크(실제로 학습되는 네트워크)
        self.critic = make_critic(observation_shape, num_actions)
        # 타깃 Q 네트워크(고정된 타깃 값 계산용)
        self.target_critic = make_critic(observation_shape, num_actions)
        # 온라인 네트워크 파라미터를 학습할 옵티마이저
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        # 스텝마다 호출할 학습률 스케줄러
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape  # 디버깅/참조용 관측 shape 저장
        self.num_actions = num_actions  # 행동 개수 저장
        self.discount = discount  # Bellman 타깃 계산에 쓰는 감가율
        self.target_update_period = target_update_period  # 타깃 갱신 주기 저장
        self.clip_grad_norm = clip_grad_norm  # grad norm 클리핑 값 저장
        self.use_double_q = use_double_q  # Double-DQN 플래그 저장

        self.critic_loss = nn.MSELoss()  # DQN의 기본 손실: TD target과 Q(s,a)의 MSE

        self.update_target_critic()  # 시작 시 target <- online으로 동기화

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Used for evaluation.
        """
        # 입력 관측을 torch 텐서로 바꾸고 배치 차원(1)을 추가한다.
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # 입실론-그리디:
        # 확률 epsilon으로 랜덤 행동(탐험), 아니면 argmax Q 행동(활용)을 선택한다.
        if np.random.rand() < epsilon:  # 탐험 확률 epsilon
            action = torch.randint(
                low=0,
                high=self.num_actions,
                size=(1,),
                device=observation.device,
            )  # 랜덤 행동 선택
        else:  # 활용할 때
            with torch.no_grad():  # 행동 선택은 역전파 대상이 아니므로 no_grad 블록에서 계산
                q_values = self.critic(observation)  # 온라인 네트워크로 Q값 계산
                action = q_values.argmax(dim=-1)  # 최대 Q값의 행동 인덱스 선택

        return ptu.to_numpy(action).squeeze(0).item()  # 파이썬 int로 변환해 반환

    def update_critic(
        self,
        obs: torch.Tensor,  # 현재 상태 배치 s_t
        action: torch.Tensor,  # 실제 수행한 행동 배치 a_t
        reward: torch.Tensor,  # 보상 배치 r_t
        next_obs: torch.Tensor,  # 다음 상태 배치 s_{t+1}
        done: torch.Tensor,  # 종료 플래그 배치 done_t
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape  # reward는 [B]여야 한다는 shape 체크 용도
        del batch_size  # 현재 코드에서는 직접 쓰지 않으므로 명시적으로 unused 처리

        # Compute target values
        with torch.no_grad():  # 타깃 값은 역전파 대상이 아니므로 no_grad 블록에서 계산
            # TODO(student): compute target values
            # next 상태의 Q값을 타깃 네트워크로 계산한다.
            next_qa_values = self.target_critic(next_obs)  # Q_target(s_{t+1}, a) 형태의 텐서

            if self.use_double_q:  # Double-DQN일 때
                # 행동 선택(argmax)은 온라인 네트워크로 수행한다.
                next_action = self.critic(next_obs).argmax(dim=-1)
            else:  # 기본 DQN일 때
                # 선택/평가를 모두 타깃 네트워크에서 한다.
                next_action = next_qa_values.argmax(dim=-1)

            # 선택된 행동의 Q값만 gather로 추출한다.
            next_q_values = torch.gather(next_qa_values, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
            # Bellman 타깃: y_t = r_t + gamma * (1-done_t) * max_a' Q_target(s_{t+1},a')
            target_values = reward + self.discount * (1 - done.float()) * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)  # 온라인 네트워크의 Q(s_t, :)
        # 데이터에 들어있는 행동 a_t 위치의 Q(s_t, a_t)를 추출한다.
        q_values = torch.gather(
            qa_values, dim=-1, index=action.long().unsqueeze(-1)
        ).squeeze(-1)
        loss = self.critic_loss(q_values, target_values)  # TD 오차 MSE

        self.critic_optimizer.zero_grad()  # 이전 스텝 gradient 초기화
        loss.backward()  # 역전파로 gradient 계산
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )  # gradient 폭주 방지를 위한 norm 클리핑
        self.critic_optimizer.step()  # 파라미터 업데이트

        self.lr_scheduler.step()  # 스케줄러 1스텝 진행

        return {
            "critic_loss": loss.item(),  # critic 손실
            "q_values": q_values.mean().item(),  # 예측 Q의 평균
            "target_values": target_values.mean().item(),  # 타깃 Q의 평균
            "grad_norm": grad_norm.item(),  # 클리핑 전/후 grad norm 로깅값
        }

    def update_target_critic(self):
        # 하드 업데이트: target 파라미터를 online 파라미터로 완전 복사한다.
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,  # 관측 배치
        action: torch.Tensor,  # 행동 배치
        reward: torch.Tensor,  # 보상 배치
        next_obs: torch.Tensor,  # 다음 관측 배치
        done: torch.Tensor,  # 종료 배치
        step: int,  # 현재 전역 학습 스텝
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        # 1) 온라인 critic을 1회 업데이트한다.
        # 2) 주기적으로 타깃 네트워크를 동기화한다.
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)  # critic 업데이트 및 통계 수집

        if step % self.target_update_period == 0:  # 타깃 네트워크 업데이트 주기
            self.update_target_critic()  # 타깃 네트워크를 온라인 네트워크로 갱신

        return critic_stats  # 로깅용 통계 반환
