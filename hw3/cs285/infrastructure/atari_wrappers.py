import numpy as np
import gym
from gym import spaces
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """FIRE 액션 전까지 화면이 멈춘 Atari 환경을 위해 reset 직후 액션을 수행한다."""
        gym.Wrapper.__init__(self, env)
        # 많은 Atari 게임은 reset 후 FIRE를 눌러야 실제 게임이 시작된다.
        # action 의미가 예상과 다르면 잘못된 래퍼 적용이므로 즉시 실패시킨다.
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        # 이 과제 코드베이스는 Gym 구 API 스타일 reset/step을 사용한다.
        # 1) 환경 reset
        # 2) FIRE 성격의 액션을 넣어 정지된 시작 화면에서 벗어남
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """보상을 부호 기준으로 {+1, 0, -1}로 클리핑한다."""
        # 보상 클리핑은 Atari-DQN의 핵심 트릭 중 하나로,
        # 게임마다 다른 보상 스케일을 안정화하고 큰 그래디언트 폭주를 줄여준다.
        return np.sign(reward)


def wrap_deepmind(env: gym.Env):
    """DeepMind 스타일 Atari 전처리 구성을 적용한다."""
    # 프레임스킵/보상클리핑 등 추가 처리 전에, 원본 환경 기준 통계를 기록한다.
    env = RecordEpisodeStatistics(env)
    # 표준 Atari 전처리
    env = AtariPreprocessing(
        env,
        noop_max=30,                 # 랜덤 no-op 시작으로 초기 상태 다양성 확보
        frame_skip=4,                # 액션 반복/시간축 다운샘플링
        screen_size=84,              # DQN 표준 입력 해상도
        terminal_on_life_loss=False, # 라이프 손실을 종료로 보지 않고 실제 게임 종료만 사용
        grayscale_obs=True,          # 스택 전 단일 채널(흑백)로 변환
    )
    # FrameStack은 단일 흑백 프레임을 (4, 84, 84) 스택 상태로 만들어,
    # 에이전트가 최근 히스토리에서 속도/움직임 정보를 추론하게 해준다.
    env = FrameStack(env, num_stack=4)
    return env
