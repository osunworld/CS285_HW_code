import os  # 파일/디렉토리 경로 처리용
import time  # 시간 측정 및 타임스탬프 생성용

from cs285.agents.pg_agent import PGAgent  # 정책경사 에이전트 클래스


import gym  # 환경 생성
import numpy as np  # 수치 계산/난수
import torch  # PyTorch
from cs285.infrastructure import pytorch_util as ptu  # GPU/CPU 유틸

from cs285.infrastructure import utils  # trajectory 수집/metric 계산 유틸
from cs285.infrastructure.logger import Logger  # 로깅 도구
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper  # 연속행동 노이즈 래퍼

MAX_NVIDEO = 2  # 텐서보드에 저장할 최대 비디오 개수


def run_training_loop(args):
    logger = Logger(args.logdir)  # 현재 실험 로그 디렉토리 연결

    # set random seeds
    np.random.seed(args.seed)  # numpy 시드 고정
    torch.manual_seed(args.seed)  # torch 시드 고정
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)  # 디바이스 초기화

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)  # 환경 생성
    discrete = isinstance(env.action_space, gym.spaces.Discrete)  # 이산/연속 행동공간 체크

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"  # 이산행동엔 노이즈 사용 불가
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)  # 연속행동에 노이즈 추가

    max_ep_len = args.ep_len or env.spec.max_episode_steps  # 에피소드 최대 길이

    ob_dim = env.observation_space.shape[0]  # 관측 차원
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]  # 행동 차원(이산이면 action 개수)

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep  # mujoco 계열 fps 계산
    else:
        fps = env.env.metadata["render_fps"]  # 일반 gym env fps 사용

    # initialize agent
    agent = PGAgent(  # 하이퍼파라미터로 PG 에이전트 생성
        ob_dim,  # 관측 차원
        ac_dim,  # 행동 차원
        discrete,  # 행동공간 타입
        n_layers=args.n_layers,  # 정책/가치망 depth
        layer_size=args.layer_size,  # hidden width
        gamma=args.discount,  # 할인율
        learning_rate=args.learning_rate,  # actor 학습률
        use_baseline=args.use_baseline,  # critic 사용 여부
        use_reward_to_go=args.use_reward_to_go,  # RTG Q 추정 여부
        normalize_advantages=args.normalize_advantages,  # advantage 정규화 여부
        baseline_learning_rate=args.baseline_learning_rate,  # critic 학습률
        baseline_gradient_steps=args.baseline_gradient_steps,  # critic 업데이트 횟수
        gae_lambda=args.gae_lambda,  # GAE lambda
    )

    total_envsteps = 0  # 지금까지 수집한 총 환경 스텝
    start_time = time.time()  # 실험 시작 시각



    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")  # 현재 iteration 출력
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        # TODO: 학습용 trajectory 수집
        total_envsteps += envsteps_this_batch  # 누적 env step 갱신

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}  # key별 trajectory 리스트로 재구성

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(trajs_dict['observation'], trajs_dict['action'],
                                        trajs_dict['reward'], trajs_dict['terminal'],
                                        )  # agent.update(...) 결과 dict
        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")  # 평가 trajectory 수집 시작
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len  # 현재 정책으로 평가 수집
            )

            logs = utils.compute_metrics(trajs, eval_trajs)  # 학습/평가 성능 통계 계산
            # compute additional metrics
            logs.update(train_info)  # 학습 loss 등 추가
            logs["Train_EnvstepsSoFar"] = total_envsteps  # 누적 스텝 로깅
            logs["TimeSinceStart"] = time.time() - start_time  # 경과 시간 로깅
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]  # 초기 데이터 수집 리턴 기록

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))  # 콘솔 출력
                logger.log_scalar(value, key, itr)  # 텐서보드 scalar 저장
            print("Done logging...\n\n")  # 로깅 완료 메시지

            logger.flush()  # 디스크로 즉시 flush

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")  # 비디오 trajectory 수집 시작
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True  # 렌더 on
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,  # trajectory 프레임
                itr,  # 현재 iteration
                fps=fps,  # 저장 fps
                max_videos_to_save=MAX_NVIDEO,  # 저장 비디오 수 제한
                video_title="eval_rollouts",  # 텐서보드 비디오 태그명
            )


def main():
    import argparse  # 인자 파서

    parser = argparse.ArgumentParser()  # 커맨드라인 파서 생성
    parser.add_argument("--env_name", type=str, required=True)  # 환경 이름
    parser.add_argument("--exp_name", type=str, required=True)  # 실험 이름
    parser.add_argument("--n_iter", "-n", type=int, default=200)  # 학습 iteration 수

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")  # RTG 사용
    parser.add_argument("--use_baseline", action="store_true")  # baseline critic 사용
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)  # critic lr
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)  # critic step 수
    parser.add_argument("--gae_lambda", type=float, default=None)  # GAE lambda
    parser.add_argument("--normalize_advantages", "-na", action="store_true")  # advantage 정규화
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)  # 할인율 gamma
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)  # actor lr
    parser.add_argument("--n_layers", "-l", type=int, default=2)  # 신경망 층 수
    parser.add_argument("--layer_size", "-s", type=int, default=64)  # hidden 크기

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)  # 랜덤 시드
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")  # GPU 비활성화 옵션
    parser.add_argument("--which_gpu", "-gpu_id", default=0)  # GPU id
    parser.add_argument("--video_log_freq", type=int, default=-1)  # 비디오 로그 주기
    parser.add_argument("--scalar_log_freq", type=int, default=1)  # scalar 로그 주기

    parser.add_argument("--action_noise_std", type=float, default=0)  # 행동 노이즈 표준편차

    args = parser.parse_args()  # 인자 파싱

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")  # 기본 데이터 폴더

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)  # 데이터 폴더 생성

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)  # 최종 로그 경로 결합
    args.logdir = logdir  # args에 logdir 저장
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)  # 실험별 로그 디렉토리 생성

    run_training_loop(args)  # 학습 루프 실행


if __name__ == "__main__":
    main()  # 스크립트 직접 실행 시 엔트리포인트
