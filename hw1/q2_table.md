| num_agent_train_steps_per_iter | Eval_AverageReturn Mean | Eval_StdReturn | Expert Mean | BC/Expert | Run |
| --- | --- | --- | --- | --- | --- |
| 500 | 2.03 | 0.23 | 5383.31 | 0.04% | q1_bc_walker2d_steps_500_Walker2d-v4_13-02-2026_14-10-58 |
| 1000 | 1034.87 | 736.75 | 5383.31 | 19.22% | q1_bc_walker2d_steps_1000_Walker2d-v4_13-02-2026_14-11-00 |
| 2000 | 5027.22 | 42.56 | 5383.31 | 93.39% | q1_bc_walker2d_steps_2000_Walker2d-v4_13-02-2026_14-11-01 |
| 3000 | 3017.20 | 2514.78 | 5383.31 | 56.05% | q1_bc_walker2d_steps_3000_Walker2d-v4_13-02-2026_14-11-02 |

**Figure Caption**: Hyperparameter: `num_agent_train_steps_per_iter` (500, 1000, 2000, 3000). Reason: this directly controls optimization effort in BC (how many gradient updates are applied on fixed expert data), so it is a natural axis to study underfitting vs. over-training while keeping model size, data source, and BC setting fixed.
