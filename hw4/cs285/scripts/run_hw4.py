import os
import time
from typing import Optional
from matplotlib import pyplot as plt
import yaml
from cs285 import envs

# 이번 스크립트에서 사용할 두 종류 에이전트:
# - ModelBasedAgent: 동역학 모델 + MPC
# - SoftActorCritic: MBPO 모드에서 정책 학습용
from cs285.agents.model_based_agent import ModelBasedAgent
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config

import argparse

from cs285.envs import register_envs

register_envs()


def collect_mbpo_rollout(
    env: gym.Env,
    mb_agent: ModelBasedAgent,
    sac_agent: SoftActorCritic,
    ob: np.ndarray,
    rollout_len: int = 1,
):
    obs, acs, rewards, next_obs, dones = [], [], [], [], []
    for _ in range(rollout_len):
        # TODO(student): collect a rollout using the learned dynamics models
        # HINT: get actions from `sac_agent` and `next_ob` predictions from `mb_agent`.
        # Average the ensemble predictions directly to get the next observation.
        # Get the reward using `env.get_reward`.
        ac = sac_agent.get_action(ob)
        pred_next_obs = []
        for i in range(mb_agent.ensemble_size):
            pred_i = mb_agent.get_dynamics_predictions(i, ob[None], ac[None])[0]  # (ob_dim,)
            pred_next_obs.append(pred_i)
        next_ob = np.mean(np.stack(pred_next_obs, axis=0), axis=0)  # (ob_dim,)
        rew, _ = env.get_reward(next_ob[None], ac[None])  # (1,)
        rew = rew[0]
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(False)

        ob = next_ob

    return {
        "observation": np.array(obs),
        "action": np.array(acs),
        "reward": np.array(rewards),
        "next_observation": np.array(next_obs),
        "done": np.array(dones),
    }


def run_training_loop(
    config: dict, logger: Logger, args: argparse.Namespace, sac_config: Optional[dict]
):
    # 학습 전체 흐름(큰 그림):
    # 1) 환경에서 데이터 수집 -> 2) 동역학 모델 학습
    # 3) (옵션) SAC 학습(MBPO) -> 4) 평가 및 로깅

    # set random seeds
    # 같은 seed면 최대한 비슷한 결과가 나오도록 난수 고정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 사용할 장치(CPU/GPU) 설정
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    # 학습용/평가용/영상용 환경을 분리하면 실험 관리가 쉬워진다.
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    # 에피소드 최대 길이(설정값이 없으면 환경 기본값 사용)
    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our MPC implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 2

    # initialize agent
    # 기본 에이전트는 모델기반 에이전트
    mb_agent = ModelBasedAgent(
        env,
        **config["agent_kwargs"],
    )
    # actor_agent는 "실제로 환경에서 행동할 주체"를 가리킨다.
    # 기본은 mb_agent, MBPO를 켜면 sac_agent로 교체된다.
    actor_agent = mb_agent

    # 실제 환경 transition을 저장하는 버퍼
    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # if doing MBPO, initialize SAC and make that our main agent that we use to
    # collect data and evaluate
    if sac_config is not None:
        # MBPO 모드: SAC를 별도로 초기화하고 정책 역할을 맡긴다.
        sac_agent = SoftActorCritic(
            env.observation_space.shape,
            env.action_space.shape[0],
            **sac_config["agent_kwargs"],
        )
        # SAC는 "실제 데이터 + 모델에서 만든 가상 데이터"를 섞어서 학습하므로
        # 별도 버퍼를 하나 더 사용한다.
        sac_replay_buffer = ReplayBuffer(sac_config["replay_buffer_capacity"])
        actor_agent = sac_agent

    # 지금까지 환경에서 모은 총 스텝 수(보고서 x축에서 자주 사용)
    total_envsteps = 0

    # 바깥 루프 1번 = "데이터 수집 + 모델학습(+SAC학습) + 평가" 한 사이클
    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
            # HINT: Use `utils.RandomPolicy` and `utils.sample_trajectories`
            random_policy = utils.RandomPolicy(env)
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env, random_policy, config["initial_batch_size"], ep_len
            )
        else:
            # TODO(student): collect at least config["batch_size"] transitions with our `actor_agent`
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env, actor_agent, config["batch_size"], ep_len
            )

        # 이번 iteration에서 모은 env step을 누적
        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        # trajectory 단위 데이터를 transition 배열 형태로 버퍼에 넣는다.
        for traj in trajs:
            replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # if doing MBPO, add the collected data to the SAC replay buffer as well
        if sac_config is not None:
            # MBPO에서는 SAC도 같은 실제 데이터를 봐야 하므로 같이 저장
            for traj in trajs:
                sac_replay_buffer.batched_insert(
                    observations=traj["observation"],
                    actions=traj["action"],
                    rewards=traj["reward"],
                    next_observations=traj["next_observation"],
                    dones=traj["done"],
                )

        # update agent's statistics with the entire replay buffer
        # 현재까지 모은 "전체 실제 데이터"로 표준화 통계를 갱신
        # (평균/표준편차가 바뀌면 모델 학습 안정성에 큰 영향)
        mb_agent.update_statistics(
            obs=replay_buffer.observations[: len(replay_buffer)],
            acs=replay_buffer.actions[: len(replay_buffer)],
            next_obs=replay_buffer.next_observations[: len(replay_buffer)],
        )

        # train agent
        print("Training agent...")
        # all_losses: 동역학 모델 학습 손실 추적용
        all_losses = []
        for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            # step_losses: 앙상블 안 각 모델 손실을 모아 평균내기 위한 리스트
            step_losses = []
            # TODO(student): train the dynamics models
            # HINT: train each dynamics model in the ensemble with a *different* batch of transitions!
            # Use `replay_buffer.sample` with config["train_batch_size"].
            # 보통 여기서는:
            # - i in ensemble_size 루프를 돌며
            # - 모델마다 다른 minibatch를 뽑아 update(i, ...) 호출
            # - 반환된 loss를 step_losses에 append
            for i in range(mb_agent.ensemble_size):
                batch = replay_buffer.sample(config["train_batch_size"])
                loss = mb_agent.update(
                    i,
                    obs=batch["observations"],
                    acs=batch["actions"],
                    next_obs=batch["next_observations"],
                )
                step_losses.append(loss)
            
            all_losses.append(np.mean(step_losses))

        # on iteration 0, plot the full learning curve
        # 첫 iteration 학습곡선을 저장해 디버깅할 때 빠르게 품질 확인
        if itr == 0:
            plt.plot(all_losses)
            plt.title("Iteration 0: Dynamics Model Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))

        # log the average loss
        # iteration 전체 평균 손실을 텐서보드에 기록
        loss = np.mean(all_losses)
        logger.log_scalar(loss, "dynamics_loss", itr)

        # for MBPO: now we need to train the SAC agent
        if sac_config is not None:
            print("Training SAC agent...")
            for i in tqdm.trange(
                sac_config["num_agent_train_steps_per_iter"], dynamic_ncols=True
            ):
                if sac_config["mbpo_rollout_length"] > 0:
                    # collect a rollout using the dynamics model
                    # 실제 버퍼에서 시작 상태 1개를 뽑아 모델 속에서 가상 rollout 생성
                    rollout = collect_mbpo_rollout(
                        env,
                        mb_agent,
                        sac_agent,
                        # sample one observation from the "real" replay buffer
                        replay_buffer.sample(1)["observations"][0],
                        sac_config["mbpo_rollout_length"],
                    )
                    # insert it into the SAC replay buffer only
                    # 가상 데이터는 실제 버퍼가 아니라 SAC 전용 버퍼에만 넣는다.
                    sac_replay_buffer.batched_insert(
                        observations=rollout["observation"],
                        actions=rollout["action"],
                        rewards=rollout["reward"],
                        next_observations=rollout["next_observation"],
                        dones=rollout["done"],
                    )
                # train SAC
                # SAC는 매 step마다 버퍼에서 batch를 꺼내 오프폴리시 업데이트
                batch = sac_replay_buffer.sample(sac_config["batch_size"])
                batch = ptu.from_numpy(batch) 
                sac_agent.update(
                    batch["observations"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_observations"],
                    batch["dones"],
                    i,
                )

        # Run evaluation
        # 평가 trajectory 수가 0이면 평가 스킵
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        # 평균 성능을 핵심 지표로 기록
        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            # trajectory가 여러 개면 분산/최댓값/최솟값도 같이 남긴다.
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)

            if args.num_render_trajectories > 0:
                # 선택적으로 평가 rollout 영상을 저장
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    actor_agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    itr,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    # 실행 인자 정의
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--sac_config_file", type=str, default=None)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    # yaml 설정 로드 + logger 생성
    config = make_config(args.config_file)
    logger = make_logger(config)

    if args.sac_config_file is not None:
        # MBPO 모드면 SAC 설정도 추가로 읽는다.
        sac_config = make_config(args.sac_config_file)
    else:
        sac_config = None

    # 실제 학습 루프 실행
    run_training_loop(config, logger, args, sac_config)


if __name__ == "__main__":
    main()
