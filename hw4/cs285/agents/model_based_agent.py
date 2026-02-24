from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    # Model-Based RL 에이전트:
    # 1) (상태,행동) -> 다음 상태 변화를 예측하는 동역학 모델을 학습하고
    # 2) 그 예측 모델로 미래를 미리 굴려(MPC) 좋은 행동을 고른다.
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        # MPC(모델 예측 제어)에 필요한 하이퍼파라미터 저장
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        # 관측/행동이 벡터(1차원) 형태인 환경만 지원한다.
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        # 상태 차원(ob_dim), 행동 차원(ac_dim) 저장
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # 여러 개의 동역학 모델(앙상블)을 만든다.
        # 이유: 모델 하나보다 여러 개 평균이 더 안정적인 예측을 주는 경우가 많다.
        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        # 앙상블 전체 파라미터를 한 optimizer로 학습
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        # 예측값과 정답의 차이를 재는 손실 함수(MSE)
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        # 표준화(normalization)를 위한 평균/표준편차를 buffer로 보관한다.
        # buffer는 모델 저장/로드 시 함께 저장되지만, gradient는 계산되지 않는다.
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        # numpy -> torch 텐서로 바꿔서 GPU/CPU(device)에서 계산 가능하게 한다.
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        eps = 1e-8  # 작은 상수로 나누기에서 0으로 나누는 것을 방지한다 
        obs_acs = torch.cat([obs, acs], dim=-1)  # (batch_size, ob_dim + ac_dim)
        obs_acs_norm = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)  # 입력을 표준화
        obs_delta = next_obs - obs  # 관측 변화량 계산
        obs_delta_norm = (obs_delta - self.obs_delta_mean) / (self.obs_delta_std + eps)  # 출력도 표준화
        # 입력을 표준화 (mean=0, std=1) 
        self.optimizer.zero_grad()
        pred_obs_delta_norm = self.dynamics_models[i](obs_acs_norm)
        loss = self.loss_fn(pred_obs_delta_norm, obs_delta_norm)
        loss.backward()
        self.optimizer.step()

        # logger가 기록하기 쉽도록 다시 numpy 스칼라로 반환
        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        # 이 함수는 "학습"이 아니라 통계량 계산이므로 gradient가 필요 없다.
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update the statistics
        # 핵심 아이디어:
        # - 입력 통계: concat([obs, acs])의 평균/표준편차
        # - 출력 통계: obs_delta = next_obs - obs 의 평균/표준편차
        # 이렇게 하면 모델이 더 안정적으로 학습된다.
        obs_acs = torch.cat([obs, acs], dim=-1)  # (n, ob_dim + ac_dim)
        obs_delta = next_obs - obs  # (n, ob_dim)
        self.obs_acs_mean = obs_acs.mean(dim=0)
        self.obs_acs_std = obs_acs.std(dim=0)
        self.obs_delta_mean = obs_delta.mean(dim=0)
        self.obs_delta_std = obs_delta.std(dim=0)
    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        eps = 1e-8
        obs_acs = torch.cat([obs, acs], dim=-1)  # (batch_size, ob_dim + ac_dim)
        obs_acs_norm = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)  # 입력을 표준화
        pred_obs_delta_norm = self.dynamics_models[i](obs_acs_norm)  # 모델 예측 (표준화된 관측 변화량)
        pred_obs_delta = pred_obs_delta_norm * (self.obs_delta_std + eps) + self.obs_delta_mean
        pred_next_obs = obs + pred_obs_delta  # 다음 관측 예측 (obs + 변화량)
        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        for acs in action_sequences.transpose(1, 0, 2):  # (horizon, mpc_num_action_sequences, ac_dim) -> (mpc_num_action_sequences, ac_dim)
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            next_obs = np.stack(
                [
                    self.get_dynamics_predictions(i, obs[i], acs)
                    for i in range(self.ensemble_size)
                ],
                axis=0,
            )  # (ensemble_size, mpc_num_action_sequences, ob_dim)
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.
            flat_next_obs = next_obs.reshape(
                self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim
            )
            flat_acs = np.tile(acs[None,:,:], (self.ensemble_size, 1, 1)).reshape(
                self.ensemble_size * self.mpc_num_action_sequences, self.ac_dim
            )
            rewards, _ = self.env.get_reward(flat_next_obs, flat_acs)
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
                if i == 0:
                    candidate_action_sequences = action_sequences
                else:
                    candidate_action_sequences = np.random.normal(
                        elite_mean,
                        elite_std,
                        size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
                    )
                    # clip to be within action space limits
                    candidate_action_sequences = np.clip(
                        candidate_action_sequences,
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )
                rewards = self.evaluate_action_sequences(obs, candidate_action_sequences)
                elite_idxs = np.argsort(rewards)[-self.cem_num_elites :]
                elites = candidate_action_sequences[elite_idxs]
                elite_mean = elites.mean(axis=0)
                elite_std = elites.std(axis=0)
                # smooth the updates using the alpha parameter
                if i == 0:
                    self.elite_mean = elite_mean
                    self.elite_std = elite_std
                else:
                    self.elite_mean = self.cem_alpha * self.elite_mean + (1 - self.cem_alpha) * elite_mean
                    self.elite_std = self.cem_alpha * self.elite_std + (1 - self.cem_alpha) * elite_std
            return self.elite_mean[0]
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
