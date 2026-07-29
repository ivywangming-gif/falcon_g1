# HumanoidVerse/FALCON Isaac Sim compatibility audit

审计对象是只读官方 FALCON checkout，固定上游为
`a967a6d8494f57777cf8d266a644ac8e45833301`。本报告不修改上游，也没有启动
simulator、Hydra task construction 或 PPO。

结论：选择 **BRANCH_S2 / STANDALONE_ISAACLAB_PORT**。HumanoidVerse 有通用
IsaacSim 入口，但 FALCON 特有的 dual-agent force locomotion 不能仅以
`+simulator=isaacsim` 替换 `+simulator=isaacgym`。

| 审计项 | 结果 | 证据与含义 |
| --- | --- | --- |
| Isaac Sim/Isaac Lab backend | 部分通过 | `humanoidverse/simulator/isaacsim/isaacsim.py` 和 `config/simulator/isaacsim.yaml` 存在；通用 articulation 配置却使用 H1 USD。 |
| FALCON exp 是否依赖 Isaac Gym 专用类 | 不通过 | force task 首行导入 `isaacgym.torch_utils`，父类还导入 `gymapi/gymtorch/gymutil`。 |
| multi-agent runner | 有条件通过 | `ppo_decoupled_wbc_ma.py` 的 PPO runner 依赖抽象 `BaseTask`，但无法绕过 simulator-specific task construction。 |
| G1 Isaac Sim/USD asset | 不通过 | G1 目录有 URDF/XML/mesh，没有 USD；现有 IsaacSim articulation 文件指向不存在的 H1 USD。 |
| force curriculum | 不通过 | curriculum 使用 `simulator.jacobian` 与 `apply_rigid_body_force_at_pos_tensor`，尚无 IsaacLab 实现。 |
| contact/root/joint/rigid-body state | 部分通过 | IsaacSim backend 暴露这些字段并将 quaternion 转为 xyzw；body ordering、jacobian 和 force write 尚未形成可验证独立契约。 |
| reward/observation 抽象性 | 不通过 | reward/obs 直接读取 simulator tensors、body indices 和 force buffers。 |
| 只换 simulator override | 不通过（未运行） | 独立环境和安装缺失；静态 imports/asset mismatch 已足以否定 config-only 迁移。 |

## S2 迁移边界

个人仓库中的 S2 只重新实现 FALCON 自身契约：29 DoF 的 lower/upper action
split（15/14）、force-conditioned observations、force curriculum、奖励、
reset/history、G1 USD asset 和 deployment contract。纯函数层不导入任何
simulator 或训练框架；Isaac Lab 只在独立环境完成验证后作为 runtime dependency。

本轮状态为 `config_resolve=NOT_RUN_ENV_MISSING`、
`task_construction=NOT_RUN_ENV_MISSING`、`PPO_STARTED=NO`。在独立环境验证、
1-env smoke、32-env capacity smoke 和 G1 ground/contact/reset/action contract
全部通过之前，不得训练。
