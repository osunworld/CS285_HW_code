from cs285.infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        # 전형적인 리플레이 버퍼로, 전이 하나를 그대로 저장한다:
        # (obs, action, reward, next_obs, done) 형태를 행 단위로 저장.
        # 이해는 쉽지만, 이미지 프레임을 스택해서 쓰는 경우 메모리 비용이 크다.
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def sample(self, batch_size):
        # 지금까지 들어온 전이들 중에서 균등 무작위 샘플링.
        # `% self.max_size`는 용량 초과 후 ring-buffer 덮어쓰기를 지원한다.
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        리플레이 버퍼에 전이 1개를 삽입한다.

        사용 예시:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        # shape 일관성을 위해 파이썬 스칼라를 numpy 배열로 변환한다.
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            # 첫 샘플이 들어올 때 지연 할당(lazy allocation).
            # 환경에서 들어온 실제 dtype/shape로 버퍼를 정확히 구성한다.
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        # ring-buffer에 현재 쓰기 위치로 저장.
        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done

        # `size`는 지금까지 삽입된 총 횟수(최대 용량보다 커질 수 있음).
        self.size += 1


class MemoryEfficientReplayBuffer:
    """
    관측을 프레임 스택으로 쓸 때 메모리를 절약하는 리플레이 버퍼.
    """

    def __init__(self, frame_history_len: int, capacity=1000000):
        # 핵심 아이디어:
        # 전이마다 "완성된 스택 관측"을 저장하지 않는다.
        # 대신 개별 프레임은 1회 저장하고, 인덱스 테이블로 스택을 재구성한다.
        self.max_size = capacity

        # obs와 next_obs 프레임을 모두 안전하게 보관하려면 이론적으로 2배 용량이 필요하다.
        # 그렇지 않으면 오래된 obs 프레임이 덮어써졌는데도,
        # next_observation_framebuffer_idcs가 그 오래된 위치를 계속 가리킬 수 있다.
        # (실제로는 미사용 데이터가 페이지아웃되어 큰 문제 없이 동작한다.)
        self.max_framebuffer_size = 2 * capacity

        self.frame_history_len = frame_history_len
        self.size = 0
        self.actions = None
        self.rewards = None
        self.dones = None

        self.observation_framebuffer_idcs = None
        self.next_observation_framebuffer_idcs = None
        self.framebuffer = None
        self.observation_shape = None

        self.current_trajectory_begin = None
        self.current_trajectory_framebuffer_begin = None
        self.framebuffer_idx = None

        self.recent_observation_framebuffer_idcs = None

    def sample(self, batch_size):
        # 전이 행을 샘플링한 뒤, 미리 계산된 인덱스로 프레임 스택을 모은다.
        # 반환되는 관측 shape은 [batch, frame_history_len, H, W]이다.
        rand_indices = (
            np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        )

        observation_framebuffer_idcs = (
            self.observation_framebuffer_idcs[rand_indices] % self.max_framebuffer_size
        )
        next_observation_framebuffer_idcs = (
            self.next_observation_framebuffer_idcs[rand_indices]
            % self.max_framebuffer_size
        )

        return {
            "observations": self.framebuffer[observation_framebuffer_idcs],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.framebuffer[next_observation_framebuffer_idcs],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def _insert_frame(self, frame: np.ndarray) -> int:
        """
        리플레이 버퍼의 프레임 저장소에 프레임 1장을 삽입한다.

        반환값은 저장된 프레임의 인덱스다.
        """
        assert (
            frame.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert frame.dtype == np.uint8, "Observation should be uint8 (0-255)"

        # 프레임은 전용 framebuffer에 저장된다.
        # 주의: 인덱스 유효성은 상위 로직이 보장한다고 가정한다.
        self.framebuffer[self.framebuffer_idx] = frame
        frame_idx = self.framebuffer_idx
        self.framebuffer_idx = self.framebuffer_idx + 1

        return frame_idx

    def _compute_frame_history_idcs(
        self, latest_framebuffer_idx: int, trajectory_begin_framebuffer_idx: int
    ) -> np.ndarray:
        """
        최신 프레임 인덱스와 에피소드 시작 인덱스를 기반으로,
        해당 시점의 프레임 히스토리를 구성하는 프레임 인덱스들을 계산한다.

        여기 인덱스는 일반 전이 버퍼가 아니라 observation framebuffer 기준이다.
        """
        # [latest-k+1, ..., latest] 인덱스를 만들고 에피소드 시작점에서 clamp한다.
        # 즉, 에피소드 시작 부분에서 과거 프레임이 부족하면
        # 해당 에피소드 첫 프레임을 반복해 패딩하는 효과가 난다.
        return np.maximum(
            np.arange(-self.frame_history_len + 1, 1) + latest_framebuffer_idx,
            trajectory_begin_framebuffer_idx,
        )

    def on_reset(
        self,
        /,
        observation: np.ndarray,
    ):
        """
        새 에피소드의 첫 관측이 들어왔을 때 호출한다.
        """
        # 일반 ReplayBuffer와의 중요한 의미 차이:
        # - on_reset은 첫 관측(프레임)만 저장한다.
        # - 아직 전이는 만들지 않는다(아직 action/reward/next_obs가 없음).
        # 전이 행은 이후 env 1스텝 뒤 insert(...)에서 생성된다.
        assert (
            observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.observation_shape is None:
            self.observation_shape = observation.shape
        else:
            assert self.observation_shape == observation.shape

        if self.observation_framebuffer_idcs is None:
            self.observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.next_observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.framebuffer = np.empty(
                (self.max_framebuffer_size, *observation.shape), dtype=observation.dtype
            )
            self.framebuffer_idx = 0
            self.current_trajectory_begin = 0
            self.current_trajectory_framebuffer_begin = 0

        # 전이 버퍼 기준으로 현재 에피소드 시작 위치를 기록한다.
        self.current_trajectory_begin = self.size

        # 첫 관측 프레임을 프레임 버퍼에 삽입한다.
        self.current_trajectory_framebuffer_begin = self._insert_frame(observation)
        # 프레임 히스토리 인덱스를 계산만 하고, 전이 행 저장은 다음 관측이 생길 때 한다.
        # 이 값은 다음 action을 위한 "현재 스택 관측" 인덱스를 의미한다.
        self.recent_observation_framebuffer_idcs = self._compute_frame_history_idcs(
            self.current_trajectory_framebuffer_begin,
            self.current_trajectory_framebuffer_begin,
        )

    def insert(
        self,
        /,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        리플레이 버퍼에 전이 1개를 삽입한다.

        사용 예시:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        # 이 메모리 효율 버퍼에서는 호출자가 다음만 넘긴다:
        # (action, reward, next_observation, done)
        # 현재 관측 스택은 on_reset/이전 스텝에서 미리 준비되어
        # `recent_observation_framebuffer_idcs`에 저장되어 있다.
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        assert (
            next_observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert next_observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.actions is None:
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.observation_shape
        assert done.shape == ()

        # 전이 1행 기록:
        # obs 스택 인덱스 + action/reward/done
        self.observation_framebuffer_idcs[
            self.size % self.max_size
        ] = self.recent_observation_framebuffer_idcs
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.dones[self.size % self.max_size] = done

        # 다음 raw 프레임을 1회 저장한다.
        next_frame_idx = self._insert_frame(next_observation)

        # 다음 관측 스택을 구성할 프레임 인덱스를 계산한다.
        # 이는 s_{t+1}에 해당하는 스택 프레임 인덱스들이다.
        next_framebuffer_idcs = self._compute_frame_history_idcs(
            next_frame_idx, self.current_trajectory_framebuffer_begin
        )
        self.next_observation_framebuffer_idcs[
            self.size % self.max_size
        ] = next_framebuffer_idcs

        self.size += 1

        # 다음 스텝에서 사용할 "현재 관측 스택"을 준비한다.
        # 이 값은 아직 샘플링 대상이 아니며, 새 에피소드가 시작되면 덮어써진다.
        # 즉, 다음 action을 위한 "현재 관측 스택"으로 갱신된다.
        self.recent_observation_framebuffer_idcs = next_framebuffer_idcs
