# Data collections

Status snapshot: **2026-08-24**. Datasets under `data/` are gitignored.

| ID | Task | Dataset path | Expert checkpoint | Episodes | Steps | Success | Status | Notes |
|----|------|--------------|-------------------|----------|-------|---------|--------|-------|
| demo-allegro-smoke | Grasp-Allegro | `data/demos/grasp_allegro_expert.zarr` | `logs/rsl_rl/allegro_grasp/2026-08-22_10-37-45_initial/model_5600.pt` | 100 | 50,000 | 100/100 | done | 64 envs, collection_steps=1000. Log: `logs/collect_allegro_demos.log`. obs_dim=115, action_dim=22 |
| demo-allegro-full | Grasp-Allegro | `data/demos/grasp_allegro_expert_full.zarr` | `logs/rsl_rl/allegro_grasp/2026-08-22_10-37-45_initial/model_5700.pt` | 2000 | 998,360 | 1979/2000 | done | 256 envs, collection_steps=4000. Log: `logs/collect_allegro_demos_full.log`. Used for `dp-allegro-full` |

## Not yet collected

- Expert demos for LEAP / Shadow / Sharpa / Wuji grasp
- Expert demos for any in-hand rotation task
- Multi-embodiment combined datasets for cross-hand diffusion

## Template (copy for new collections)

```md
| id | Task | path | checkpoint | n_episodes | n_steps | n_success | status | notes |
```
