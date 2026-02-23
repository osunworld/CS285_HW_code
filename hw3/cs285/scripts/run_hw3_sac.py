import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config

import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # ------------------------------------------------------------
    # 1) 실험 재현성 + 디바이스 설정
    # ------------------------------------------------------------
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # ------------------------------------------------------------
    # 2) 환경 생성: 학습/평가/렌더를 분리해서 사용
    # ------------------------------------------------------------
    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    # 한 에피소드 최대 길이와 SGD 배치 크기 결정
    # - ep_len: yaml에서 지정하지 않으면 환경 기본값 사용
    # - batch_size: 매 update마다 replay buffer에서 뽑을 transition 수
    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    # SAC 구현은 현재 연속 행동공간만 지원한다.
    # (이산 행동은 DQN 파이프라인 사용)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    # 관측/행동 차원은 네트워크 생성 시 필수
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # ------------------------------------------------------------
    # 3) 에이전트/리플레이버퍼 초기화
    # ------------------------------------------------------------
    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    # off-policy 핵심 구성요소:
    # 환경에서 모은 transition을 저장해두고 랜덤 샘플링해 학습한다.
    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # 현재 state(observation) 시작값
    observation = env.reset()

    # ------------------------------------------------------------
    # 4) 메인 학습 루프
    # ------------------------------------------------------------
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        # warmup 단계에서는 랜덤 행동으로 데이터 다양성을 확보
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            # 학습된 정책(actor)로 현재 관측에서 행동 샘플링
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            # RecordEpisodeStatistics wrapper가 episode return/length를 info에 넣어준다.
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        # --------------------------------------------------------
        # 5) 업데이트 구간
        # --------------------------------------------------------
        # Train the agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            # batch에는 보통 obs/acts/rews/next_obs/dones가 들어간다.
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)
            # TODO(student): agent.update(...) 호출
            # - critic 업데이트(bootstrapping)
            # - actor 업데이트(E[Q] 최대화)
            # - target critic 업데이트(hard/soft)
            update_info = agent.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
                step=step,
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # --------------------------------------------------------
        # 6) 주기적 평가(탐험 없는 정책 성능 확인)
        # --------------------------------------------------------
        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    # CLI 인자: config 파일 + 로깅/평가 주기 + 디바이스 옵션
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    # yaml 설정 로드 + 로그 디렉토리 생성
    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    # 실제 학습 루프 시작
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
