# DAgger Results (Ant-v4, Walker2d-v4)

## Ant-v4
- Figure: `dagger_Ant_v4_curve.png`
- DAgger run: `q2_dagger_ant_curve_Ant-v4_13-02-2026_14-37-36`
- BC baseline run: `q1_bc_ant_Ant-v4_13-02-2026_11-02-55`

Caption: Task=`Ant-v4`; network=`MLP (n_layers=2, size=64)`; expert data=`cs285/expert_data` (same source as BC), DAgger uses initial expert data at itr=0 and adds policy rollouts relabeled by expert each iteration; n_iter=10, batch_size=1000, train_batch_size=100, num_agent_train_steps_per_iter=1000, eval_batch_size=5000, learning_rate=5e-3, seed=1.

## Walker2d-v4
- Figure: `dagger_Walker2d_v4_curve.png`
- DAgger run: `q2_dagger_walker2d_curve_Walker2d-v4_13-02-2026_14-37-57`
- BC baseline run: `q1_bc_Walker2d_Walker2d-v4_13-02-2026_11-05-09`

Caption: Task=`Walker2d-v4`; network=`MLP (n_layers=2, size=64)`; expert data=`cs285/expert_data` (same source as BC), DAgger uses initial expert data at itr=0 and adds policy rollouts relabeled by expert each iteration; n_iter=10, batch_size=1000, train_batch_size=100, num_agent_train_steps_per_iter=1000, eval_batch_size=5000, learning_rate=5e-3, seed=1.
