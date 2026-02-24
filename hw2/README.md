# CS285 HW2 README

이 README는 HW2 코드 실행/재현용으로 작성되었습니다.
요구사항에 맞게 `cs285` 폴더 구조는 원본 숙제와 동일하게 유지했고,
실험 로그(`data`)는 제출에서 제외하는 것을 권장합니다.

## 1) 제출/폴더 구조 안내

- 제출 핵심: `hw2/cs285/` 내부의 `.py` 파일들 (원본과 동일한 경로/이름 유지)
- 추가 스크립트: `hw2/plot_*.py` (보고서 figure 재현용)
- **제외 권장**:
  - `hw2/data/` (대용량 로그)
  - `__pycache__/`
  - `.DS_Store`

## 2) 구현한 주요 파일

HW2 TODO 구현 대상 파일:

- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/infrastructure/utils.py`

## 3) 환경 설정

`hw2` 폴더에서 실행:

```bash
cd hw2
pip install -r requirements.txt
pip install -e .
```

TensorBoard가 없으면:

```bash
pip install tensorboard
```

## 4) 실험 실행 명령 (재현용)

아래 명령은 `hw2` 디렉토리 기준입니다.

### 4.1 CartPole PG (Small/Large batch 비교, Q2)

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

### 4.2 HalfCheetah baseline / bgs / blr 비교 (NN baseline 섹션)

```bash
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --use_baseline --exp_name sun_cheetah_baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --use_baseline --baseline_gradient_steps 1 --exp_name sun_cheetah_baseline_bgs1
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --use_baseline --baseline_learning_rate 1e-3 --exp_name sun_cheetah_baseline_blr1e3
```

Optional (`-na`) 비교:

```bash
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --use_baseline -na --exp_name cheetah_baseline
```

### 4.3 GAE lambda sweep (LunarLander)

```bash
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --exp_name lunar_lander_lambda0
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --exp_name lunar_lander_lambda1
```

### 4.4 InvertedPendulum default vs tuned (5 seed)

Default:

```bash
for seed in $(seq 1 5); do
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 --exp_name pendulum_default_s$seed -rtg --use_baseline -na --batch_size 5000 --seed $seed
done
```

Tuned (본 저장소에서 사용한 세팅 예시):

```bash
for seed in $(seq 1 5); do
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --exp_name pendulum_tune_s$seed -n 100 --seed $seed --discount 0.9 -rtg -na --gae_lambda 0.99 -l 4 -s 80 --batch_size 5000
done
```

### 4.5 Humanoid (장시간 실험)

```bash
python cs285/scripts/run_hw2.py \
  --env_name Humanoid-v4 --ep_len 1000 \
  --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
  --baseline_gradient_steps 50 \
  -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
  --exp_name humanoid --video_log_freq 5
```

## 5) Figure/Table 생성 명령

## 5.1 CartPole figure (small/large batch)

```bash
python plot_PG.py
```

출력:

- `hw2_small_batch.png`
- `hw2_large_batch.png`

## 5.2 NNB figure (baseline loss / eval return)

기본 baseline 매치:

```bash
python plot_NNB.py --env_name HalfCheetah-v4 --data_dir data --match baseline --smooth_window 1 --out_baseline hw2_nnb_baseline_loss.png --out_return hw2_nnb_eval_return.png
```

특정 실험 3개 비교:

```bash
python plot_NNB.py \
  --env_name HalfCheetah-v4 \
  --data_dir data \
  --experiments "sun_cheetah_baseline,sun_cheetah_baseline_bgs1,sun_cheetah_baseline_blr1e3" \
  --out_baseline hw2_nnb_bgs_blr_base_loss.png \
  --out_return hw2_nnb_bgs_blr_base_eval_return.png
```

## 5.3 GAE sweep figure

```bash
python plot_GAE.py \
  --env_name LunarLander-v2 \
  --data_dir data \
  --exp_prefix lunar_lander_lambda \
  --lambdas "0,0.95,0.98,0.99,1" \
  --smooth_window 21 \
  --clip_percentile 3 \
  --out hw2_lunarlander_gae_curve.png
```

## 5.4 Hyperparameter 비교 figure (Pendulum)

```bash
python plot_Hyper.py \
  --env_name InvertedPendulum-v4 \
  --data_dir data \
  --baseline_exps "pendulum_default_s1,pendulum_default_s2,pendulum_default_s3,pendulum_default_s4,pendulum_default_s5" \
  --tuned_exps "pendulum_tune_s1,pendulum_tune_s2,pendulum_tune_s3,pendulum_tune_s4,pendulum_tune_s5" \
  --out hw2_pendulum_hyper_compare.png
```

## 5.5 Humanoid learning curve figure

```bash
python plot_Humanoid.py \
  --data_dir data \
  --env_name Humanoid-v4 \
  --exp_name humanoid \
  --metric Eval_AverageReturn \
  --smooth_window 5 \
  --out hw2_humanoid_learning_curve.png
```

## 6) 참고 사항

- `data/` 로그는 실행 시 자동 생성됩니다.
- 로컬/서버 환경마다 런타임과 곡선 변동이 다를 수 있습니다(특히 seed 민감).
- matplotlib 스타일 관련 에러가 나면 현재 스크립트들은 fallback 스타일을 사용하도록 작성되어 있습니다.
