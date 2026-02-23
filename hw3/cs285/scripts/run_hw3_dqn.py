import argparse  # 커맨드라인 인자 파싱

from cs285.agents.dqn_agent import DQNAgent  # 우리가 구현한 DQN 에이전트
import cs285.env_configs  # 설정 모듈(동적 import 트리거용)

import gym  # 강화학습 환경 라이브러리
import numpy as np  # 수치 연산/배열 처리
import torch  # 텐서/신경망 학습
from cs285.infrastructure import pytorch_util as ptu  # numpy<->torch 변환, 디바이스 초기화
import tqdm  # 진행바 표시

from cs285.infrastructure import utils  # 평가 rollout 수집 함수
from cs285.infrastructure.logger import Logger  # 텐서보드 로깅 유틸
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer  # 버퍼 구현

from scripting_utils import make_logger, make_config  # config/logger 생성 헬퍼

MAX_NVIDEO = 2  # 현재 스크립트에서는 직접 사용하지 않지만 템플릿 호환을 위해 유지


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)  # numpy 난수 시드 고정(재현성)
    torch.manual_seed(args.seed)  # torch 난수 시드 고정(재현성)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)  # 사용할 디바이스 초기화

    # make the gym environment
    env = config["make_env"]()  # 학습용 환경
    eval_env = config["make_env"]()  # 수치 평가용 환경(학습과 분리)
    render_env = config["make_env"](render=True)  # 영상 저장용 렌더 환경
    exploration_schedule = config["exploration_schedule"]  # step별 epsilon 스케줄
    discrete = isinstance(env.action_space, gym.spaces.Discrete)  # DQN은 이산 행동만 지원

    assert discrete, "DQN only supports discrete action spaces"  # 연속 행동이면 즉시 중단

    agent = DQNAgent(
        env.observation_space.shape,  # 관측 shape
        env.action_space.n,  # 행동 개수
        **config["agent_kwargs"],  # critic/optimizer/discount 등 설정 전달
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):  # MuJoCo류 환경이면 physics timestep 사용
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:  # gym metadata에 렌더 fps가 있으면 사용
        fps = env.env.metadata["render_fps"]
    else:  # 정보가 없으면 기본 fps
        fps = 4

    ep_len = env.spec.max_episode_steps  # 환경에서 허용하는 최대 에피소드 길이

    observation = None  # 현재 상태 관측(루프 안에서 업데이트)

    # Replay buffer
    # 관측 shape에 따라 메모리 효율 버퍼(이미지) vs 일반 버퍼(벡터)를 선택한다.
    if len(env.observation_space.shape) == 3:
        stacked_frames = True  # Atari처럼 (C,H,W) 형태 스택 프레임 관측
        frame_history_len = env.observation_space.shape[0]  # 채널축=프레임 스택 길이
        assert frame_history_len == 4, "only support 4 stacked frames"  # 템플릿 가정
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )  # 프레임 중복 저장을 피하는 버퍼
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False  # CartPole 같은 1D 벡터 관측
        replay_buffer = ReplayBuffer()  # 일반 (s,a,r,s',done) 버퍼
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )  # 예상하지 못한 관측 형태는 에러 처리
    _ = stacked_frames  # 현재 로직에서는 직접 사용하지 않지만 가독성을 위해 변수 유지

    def reset_env_training():
        nonlocal observation  # 바깥 함수의 observation을 수정하기 위해 선언

        observation = env.reset()  # 새 에피소드 시작 관측 가져오기

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)  # LazyFrames 등도 numpy 배열로 통일

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # 메모리 효율 버퍼는 에피소드 시작 프레임을 먼저 알려줘야 한다.
            replay_buffer.on_reset(observation=observation[-1, ...])  # 가장 최근 raw frame 1장

    reset_env_training()  # 학습 루프 시작 전 초기 reset

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):  # 전역 step 루프
        epsilon = exploration_schedule.value(step)  # 현재 step의 탐험 확률

        # TODO(student): Compute action
        # 입실론-그리디 정책으로 행동 선택(탐험/활용 자동 처리)
        action = agent.get_action(observation, epsilon)  # 현재 관측과 탐험 확률로 행동 선택

        # TODO(student): Step the environment
        # 선택한 행동을 환경에 적용해 다음 전이 정보를 받는다.
        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)  # 관측 타입을 numpy로 통일
        truncated = info.get("TimeLimit.truncated", False)  # 시간제한으로 끝난 종료인지 확인

        # TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            # 중요 개념:
            # 메모리 효율 버퍼는 현재 obs 스택 인덱스를 내부 상태로 이미 들고 있어
            # insert 시에는 next frame만 넣어도 (s,a,r,s',done) 전이를 재구성할 수 있다.
            replay_buffer.insert(
                action=action,
                reward=reward,
                next_observation=next_observation[-1, ...],
                done=done and not truncated,
            )
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done and not truncated,
            )

        # Handle episode termination
        if done:  # 환경이 종료되면
            reset_env_training()  # 다음 에피소드 시작

            # RecordEpisodeStatistics 래퍼가 넣어준 에피소드 통계를 학습 로그에 기록
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation  # 종료가 아니면 상태를 한 스텝 전진

        # Main DQN training loop
        if step >= config["learning_starts"]:  # 워밍업 스텝 이후에만 파라미터 업데이트
            # TODO(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])  # 버퍼에서 미니배치 샘플링

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)  # dict 안의 ndarray를 디바이스 텐서로 변환

            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(
                obs=batch["observations"],
                action=batch["actions"],
                reward=batch["rewards"],
                next_obs=batch["next_observations"],
                done=batch["dones"],
                step=step,
            )  # critic 업데이트 + 필요시 target 업데이트

            # Logging code
            update_info["epsilon"] = epsilon  # 현재 탐험 비율 기록
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]  # 현재 학습률 기록

            if step % args.log_interval == 0:  # 지정 간격마다 텐서보드로 flush
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:  # 지정 간격마다 평가 rollout 수행
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,  # 평가 환경
                agent,  # 현재 정책(DQN 에이전트)
                args.num_eval_trajectories,  # 평가 에피소드 수
                ep_len,  # 최대 길이
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]  # 에피소드 리턴
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]  # 에피소드 길이

            logger.log_scalar(np.mean(returns), "eval_return", step)  # 평균 리턴
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)  # 평균 길이

            if len(returns) > 1:  # 다중 trajectory일 때 분산 통계도 함께 기록
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:  # 영상 저장 옵션이 켜져 있으면
                video_trajectories = utils.sample_n_trajectories(
                    render_env,  # 렌더 가능한 환경
                    agent,  # 동일 정책
                    args.num_render_trajectories,  # 영상 에피소드 수
                    ep_len,  # 최대 길이
                    render=True,  # 이미지 프레임 수집
                )

                logger.log_paths_as_videos(
                    video_trajectories,  # 저장할 trajectory들
                    step,  # 현재 step을 파일명/스텝 축에 반영
                    fps=fps,  # 프레임레이트
                    max_videos_to_save=args.num_render_trajectories,  # 저장 최대 개수
                    video_title="eval_rollouts",  # 텐서보드 비디오 탭 제목
                )


def main():
    parser = argparse.ArgumentParser()  # 인자 파서 생성
    parser.add_argument("--config_file", "-cfg", type=str, required=True)  # 설정 파일 경로

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)  # 평가 주기
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)  # 평가 개수
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)  # 영상 개수

    parser.add_argument("--seed", type=int, default=1)  # 랜덤 시드
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")  # GPU 비활성화 플래그
    parser.add_argument("--which_gpu", "-gpu_id", default=0)  # GPU id
    parser.add_argument("--log_interval", type=int, default=1000)  # 학습 로그 주기

    args = parser.parse_args()  # 실제 CLI 인자 파싱

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

    config = make_config(args.config_file)  # 설정 파일 로드 및 dict 생성
    logger = make_logger(logdir_prefix, config)  # 로그 디렉토리/라이터 초기화

    run_training_loop(config, logger, args)  # 학습 루프 실행


if __name__ == "__main__":  # 스크립트 직접 실행 시 엔트리포인트
    main()  # 메인 함수 호출
