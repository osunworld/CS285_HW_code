# CS285 HW3 README

이 문서는 HW3 제출/재현용 안내입니다.
요구사항에 맞게 `hw3/cs285` 폴더의 `.py` 파일들은 원본 숙제 레포와 동일한 이름/경로 구조를 유지합니다.

## 1) 제출 구조

- 제출 핵심: `hw3/cs285/` 전체 `.py` 파일
- 실험 설정: `hw3/experiments/`
- 재현 스크립트: `hw3/plot_*.py`
- **제출에서 제외 권장**:
  - `hw3/data/` (학습 로그)
  - `hw3/__pycache__/`
  - 생성된 대용량 비디오/이미지 로그

## 2) 주요 구현 파일

- `cs285/agents/dqn_agent.py`
- `cs285/scripts/run_hw3_dqn.py`
- `cs285/agents/soft_actor_critic.py`
- `cs285/scripts/run_hw3_sac.py`

## 3) 환경 설정

아래 명령은 `hw3` 폴더에서 실행합니다.

```bash
cd hw3
pip install -r requirements.txt
pip install -e .
```

서버에서 import 문제를 피하려면 실행은 **항상 모듈 방식**으로 권장합니다.

- 사용 권장: `python -m cs285.scripts.run_hw3_dqn ...`
- 사용 권장: `python -m cs285.scripts.run_hw3_sac ...`

## 4) 로그 경로

`run_hw3_dqn.py`, `run_hw3_sac.py`는 실행 시 자동으로 `hw3/data/` 아래에 로그 디렉토리를 생성합니다.

예시:
- `hw3/data/hw3_dqn_...`
- `hw3/data/hw3_sac_...`

## 5) 실험 실행 명령

### 5.1 DQN: CartPole 기본 + lr=0.05

```bash
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/cartpole.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/cartpole_lr005.yaml --seed 1
```

### 5.2 DQN vs Double DQN (LunarLander, 3 seeds)

Vanilla DQN:

```bash
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander.yaml --seed 2
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander.yaml --seed 3
```

Double DQN:

```bash
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3
```

### 5.3 MsPacman (DQN)

```bash
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/mspacman.yaml
```

GPU 지정 예시:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/mspacman.yaml --which_gpu 0
```

### 5.4 DQN Hyperparameter (LunarLander, 4 configs)

```bash
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/hyperparameters/lunar_hp_lr1e3.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/hyperparameters/lunar_hp_lr3e4.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/hyperparameters/lunar_hp_lr3e3.yaml --seed 1
python -m cs285.scripts.run_hw3_dqn -cfg experiments/dqn/hyperparameters/lunar_hp_lr1e3_epsfix.yaml --seed 1
```

### 5.5 SAC sanity checks

```bash
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/sanity_pendulum.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml --seed 1
```

### 5.6 SAC HalfCheetah (REINFORCE-1 / REINFORCE-10 / Reparametrize)

```bash
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/halfcheetah_reinforce1.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/halfcheetah_reinforce10.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/halfcheetah_reparametrize.yaml --seed 1
```

### 5.7 SAC Hopper (single / double / clipped)

```bash
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/hopper.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/hopper_doubleq.yaml --seed 1
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/hopper_clipq.yaml --seed 1
```

### 5.8 SAC Humanoid (best setting)

```bash
python -m cs285.scripts.run_hw3_sac -cfg experiments/sac/humanoid.yaml --seed 1
```

## 6) Figure 재현 명령

### 6.1 LunarLander: Basic DQN vs Double DQN

```bash
python plot_lunar_basic_vs_doubleq.py
```

출력:
- `hw3_lunar_basic_vs_doubleq.png`

### 6.2 MsPacman: train_return vs eval_return

```bash
python plot_mspacman.py
```

출력:
- `hw3_mspacman_train_vs_eval.png`

### 6.3 LunarLander Hyperparameter 비교

```bash
python plot_hyper.py
```

출력:
- `hw3_lunar_hyper_eval_return.png`

### 6.4 Humanoid: HW2 vs HW3 비교

```bash
python plot_humanoid_hw2_vs_hw3.py \
  --hw2_data ../hw2/data \
  --hw3_data ./data \
  --hw2_match q2_pg_humanoid \
  --hw3_match humanoid \
  --smooth 3
```

출력:
- `hw3_humanoid_hw2_vs_hw3.png`

### 6.5 REINFORCE/REPARAM 및 Hopper 추가 비교 (TensorBoard)

현재 저장소에는 해당 비교 전용 plot 스크립트가 없으므로 TensorBoard로 재현합니다.

```bash
tensorboard --logdir data --port 6006
```

브라우저에서 다음 scalar를 같은 축에 overlay:
- `eval_return`
- `q_values` (Hopper single/double/clipped 비교)

## 7) 서버 실행 메모

- 여러 GPU를 쓸 때는 프로세스별로 `CUDA_VISIBLE_DEVICES`를 분리하세요.
- 긴 학습은 `tmux`/`screen` 사용을 권장합니다.
- 제출 압축 전에 `hw3/data`와 비디오 로그를 제거하면 용량을 크게 줄일 수 있습니다.
