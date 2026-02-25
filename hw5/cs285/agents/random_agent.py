from typing import Tuple
import numpy as np


class RandomAgent:
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        # `observation_shape`는 랜덤 에이전트에서는 실제로 사용되지 않지만,
        # 다른 에이전트들과 인터페이스를 맞추기 위해 생성자 인자로 받는다.
        # (예: DQN/RND 에이전트는 관측 차원을 이용해 네트워크를 만든다.)
        self.num_actions = num_actions

    def get_action(self, *args, **kwargs):
        # TODO(student): Return a random action
        # 이 함수는 "현재 상태와 무관하게" 행동을 무작위로 하나 고른다.
        #
        # 구현 목표:
        # - 가능한 이산 행동 인덱스 [0, num_actions-1] 중 하나를 균등확률로 샘플링
        # - 반환 타입은 파이썬 int 또는 numpy 정수형이면 충분
        #
        # 왜 이렇게 하나?
        # - Problem 초반의 데이터 수집 단계에서 랜덤 정책이 baseline 역할을 한다.
        # - RND가 없을 때 순수 무작위 탐험이 어느 정도 성능을 내는지 비교 가능하다.
        #
        # 대표 구현 예시(직접 작성):
        # - np.random.randint(self.num_actions)
        #
        # 주의:
        # - 상한을 `self.num_actions - 1`로 주면 마지막 행동이 빠질 수 있다.
        #   np.random.randint(high)는 high "미포함"이다.
        # - 연속 행동 환경이 아니라 "이산 행동" 환경을 가정한 구현이다.
        return int(np.random.randint(self.num_actions))
    
    def update(self, *args, **kwargs):
        # Update is a no-op for the random agent
        # 랜덤 에이전트는 학습할 파라미터가 없어서 update 단계가 필요 없다.
        # 학습 에이전트와 동일한 호출 인터페이스를 유지하기 위해 빈 dict를 반환한다.
        return {}
