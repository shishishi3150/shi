# 异构 USV–AUV 岛礁协同围捕与阻截

当前工程：**2 USV + 3 AUV、五个独立决策智能体、物理联合动作 `[5,2]`**。MATD3 使用单帧 MLP、同类参数共享、双多头 Critic、7步回报及实际执行动作 Replay。USV 为 CyberShip II 3-DOF，AUV 为独立欠驱动水平面3-DOF；地图55×55 m、艇体、动力学、每步5积分子步和成功判据沿用原任务。

2026-11-10 当前版本为 `force_control_local_information_capture_mlp_pursuit_v2`，默认新输出根目录 `outputs/sustained_encirclement_v5`。I1 仍为 **Maneuverability-aware Velocity Obstacle Guidance**，方法标识 `maneuverability_aware_vo_v1`。旧模型、实验和失败轨迹保留，旧奖励模型须配原源码；新版本拒绝静默混用。本轮奖励契约见下文；修改前备份位于工程旁的shi_backups目录。历史 I1 迁移见 [VO 报告](validation/vo_migration_20261107_200525/REPORT.md)。

I1 用相对速度碰撞锥、平移 VO 边界与自身速度圆的解析交点选择安全航迹方向，保留实际侧滑 `chi=psi+atan2(v,u)`。名义动作只执行一个控制步的原动力学算术来估计方向，无冲突不修改动作；有冲突时用角色独立的有界航迹跟踪适配产生偏航力矩，必要时减速。无安全方向或响应不足时明确记录不可行并制动。原长预测、备用轨迹认证和执行器网格枚举已从生产代码移除，真实连续扫掠碰撞检测完整保留。

VO 对每艇只使用本艇、已投递伙伴消息和合法目标估计，不读隐藏伙伴/红船真值；保留垂向分离。环境、Target、Actor各价值头、I3 reject-only及独立追围使用同一接口。Actor straight-through 是近似梯度，VO 本身不可微。I2 信息年龄/不确定性算法与 I3 历史完整联合动作算法保持原样。论文 DOI `10.1109/TCST.2024.3375023` 的正最低速度和严格速度优势等条件在本项目不成立，因此有限场景通过不是全局安全保证。

## 当前任务与奖励（2026-11-10 sustained v5）

首次真实围捕仍要求三艘AUV距目标2–5 m、最大角隙≤160°、目标处于三角形内且边裕度≥0.25 m、三艇有效跟踪及安全条件，连续保持3 s。该事件只触发红船撤退并给每艘AUV一次+5，不结束任务。`capture_success/capture_triggered`表示这个历史事件；`success/full_mission_success/cooperative_task_success`只表示完整任务成功。

捕获后的任务终点为**累计650步**。完整成功要求首次围捕≤500步、撤退后严格围捕保持率≥**60%（用户最新指定）**、最长连续脱围≤8 s、最后连续5 s严格围捕、无严重事故，并要求两艘USV各自客观跟随质量均值≥0.35、安全有效追踪比例≥80%。安全有效追踪要求合法信息、中心距离4–12 m、真实船体净空≥0.5 m；后方表现另由跟随质量评价，不重复设半平面硬门。参数均为工程初值。未围捕的回合仍允许搜索至900步并超时失败；650步后才首次围捕也不能成为完整成功。任务结束未通过全部门槛记`mission_failure`。

奖励方案` sustained_encirclement_v5 `，所有参数位于`configs/scene_template.yaml`的`reward.engineering_initial`。旧的符号+2/−1、群体中心、独立角度/队形/保持、发现及原地转向项均不再支付，按本次指定公式统一为以下组件：

| 组件 | 初始公式/系数 |
| --- | --- |
| AUV进度 `auv_approach` | 下述折扣一致有界势函数，正收益每步≤0.05 |
| AUV质量 `auv_quality` | 首次围捕前0.06Q，后0.18Q |
| AUV速度 `auv_velocity` | 前0.02、后0.06，乘exp(−区间误差/1.5)及exp(−(速度向量误差/0.25)²) |
| AUV脱围 `auv_loss` | 前0；后−0.10(1−Q) |
| AUV安全 `auv_safety` | −f(最危险有效净空) |
| USV进度 `usv_approach` | 仅进攻阶段，目标区间4–6 m，同下述进度机制 |
| USV阻截 `usv_block` | 仅进攻阶段，新鲜有效审计0.5clip(Shapley/0.2,−1,1)，一次结算 |
| USV跟随 `usv_follow` | 仅撤退阶段，0.18Qf−0.10(1−Qf) |
| USV安全 `usv_safety` | −2max(当前风险、预测风险) |
| AUV/USV时间 | 每艇每步−0.006 |
| 首次围捕 `auv_capture` | 仅AUV一次+5 |
| 完整成功/失败 | 全体活跃艇各+100/−100，一次结算，不重复叠加失败类别 |

Q=exp(−三艇最大2–5m区间误差/1.5)×[0.5exp(−(角隙RMS误差/30°)²)+0.5clip((有符号三角形边裕度+0.5)/0.75,0,1)]。退化三角形凸包分为0。Qf=exp(−dist(d,[4,8])/4)×[0.8(1−cosβ)/2+0.2exp(−(速度向量误差/0.3)²)]。撤退方向使用合法速度估计，0.75 s平滑；速度低于0.05 m/s保留最近方向，无可靠速度则不发跟随正奖励。速度估计还需速度协方差迹≤2。

**进度公式的必要修正：**用户提出的线性截断进度在γ=0.99下存在微小折扣循环收益；估计更新、侧滑超出标称纵向步距及截断也会破坏闭环。采用Phi=A exp(−e/1.5)，A=0.05×1.5/(当前类型v_max×dt)，两个运动端点使用同一份当前合法估计。令Phi_old_frozen为旧位置在这份估计下的势、Phi_prev为上次记账势：r=min(0.05, γPhi_new−Phi_prev−γmax(0,Phi_old_frozen−Phi_prev))。无信息时不领取正向进度，维持势的折扣成本；终局Phi_new=0，USV进入撤退阶段前也清零。上式≤γPhi_new−Phi_prev，因此闭合状态循环的折扣进度不能为正。不截掉必要的负向势结算，所以它可以超过线性公式原先的−0.05下限；仍有明确有限边界。每步标称位移从配置计算（当前AUV0.075m、USV0.06m）。

为限制静态正质量分长期累计，**每艇每局过程正收益预算60**，跨捕获阶段共用，不在捕获时重置。预算不足时按比例缩放当步正过程项；时间、安全、脱围等负项完整保留，+5及终局±100不进入预算。日志同时保留原始过程项、实付项及预算余量。失败回合即使刷满预算并取得首次奖金，其单艇总回报仍≤−35（USV≤−40）；实际固定轨迹还需验证成功轨迹得分高于失败轨迹。

安全f(d)：d≥3为0；1≤d<3为(3−d)/4；0<d<1为1−d/2；接触时为1。使用船体净空并保留垂向分离。USV–红船风险只使用合法目标估计与红船保守外接圆，不读取隐藏目标状态产生导航奖励；采用1.5/0.5 m连续风险及3 s相对运动预测最近净空，和其他当前风险取最大，不叠加无界处罚。反事实审计沿用原四分支Shapley，不把APF自行绕礁当艇的贡献。

**信息边界：**正向目标奖励来自合法接收估计，保留原I2时延/丢包/方差门。`objective_quality`、严格围捕门、USV客观跟随和完整成功判据使用环境真值，仅供任务评价，绝不写入Actor/I1/I2/I3观测；CTDE Critic仍沿用既有状态。每局重置奖励预算、方向记忆和事件账本；snapshot/evaluate保持纯函数，settle一次提交。

CSV保留`team_return`（每艇均值）等原字段，并新增`episode_return`（同每艇均值）、`episode_return_team_sum`、`episode_length`、`capture_triggered`、`full_mission_success`、保持率/真实Q/最大脱围/末5秒比例、碰撞数、Shapley效果及两艇跟随质量。JSONL的`usv_follow_quality`为两艇数组，CSV同名为两艇平均并另列usv1/usv2字段。奖励组件列按v5重新生成，禁止拼接旧CSV；schema升级为`shi_experiment_data_sustained_v5`。选模按完整成功率、保持率、安全失败率、首次围捕率、折扣回报排序。

物理可达诊断采用原动力学及原APF撤退，理想真值控制器仅用于固定初始状态的可达性上界检查；不是Actor输入、人工奖励目标、课程学习或模型性能证据。初始相对编队只存在于诊断脚本。原APF抵达撤退目标附近会绕转，没有修改其脚本来降低任务难度。

## 配置与实验

| 实验 | I1 | I2 | I3 |
| --- | --- | --- | --- |
| Base | 关 | 关 | 关 |
| Full | 开 | 开 | 开 |
| w_o_I1 | 关 | 开 | 开 |
| w_o_I2 | 开 | 关 | 开 |
| w_o_I3 | 开 | 开 | 关 |
| I1_only | 开 | 关 | 关 |
| I2_only | 关 | 开 | 关 |
| I3_only | 关 | 关 | 开 |

另保留 `Traditional_CV / Random_History / No_State_Matching / No_Safety_Screening / Hard_Max`，共13预设。`No_Safety_Screening` 只关闭I3历史候选安全筛选。Base使用原共用传感与发送端局部滤波，不调用接收端I2补偿。所有组共用任务、奖励、网络及学习设置。

训练入口默认 `configs/fast.yaml`，I3 search_pool/candidates=256/4；`configs/default.yaml` 保留默认预算，`fast_economy.yaml` 仍有效。未指定 `--steps` 或 `--episodes` 时，默认运行100000个环境交互步（包含前10000步随机热身）；显式预算参数覆盖默认值。batch256、warmup10000、Replay20000、gamma0.99、n-step7、Critic每环境步更新、Actor延迟2均未减少。I1配置单独hash，运行manifest记录方法、配置hash、源码hash。独立 `counterfactual.horizon_steps` 是原红船阻截审计参数，和VO无关。

## 检查与复测

在工程根目录使用已有 Python 环境，依赖见 `requirements.txt`：

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B -m scripts.validate_sustained_reward --output validation/sustained_v5_NEW
python check_config.py --experiment Full --device cpu
python -B -m scripts.validate_vo --output validation/vo_acceptance_NEW
python -B -m scripts.validate_scene --experiment Full --red-episodes 1 --rule-episodes 6 --ideal-episodes 0 --random-episodes 3 --output validation/vo_scene_NEW
python -B -m scripts.benchmark_vo --output validation/vo_benchmark_NEW.json --fixture validation/vo_fixture_NEW.npz --repeats 24
```

验收和基准脚本拒绝覆盖已有证据。固定/随机回归保留所有失败轨迹；blind_red没有合法红船信息、initial overlap已经触礁，不能伪装为安全。性能数据分开报告Numba启动/编译与稳定运行，完整更新样本覆盖Actor延迟的两个阶段，CPU测量不能当作云GPU吞吐。

短实现烟雾检查（缩小网络/batch，仅检查链路，不作为性能证据）：

```powershell
python -B -m scripts.train --experiment Full --seed 6200 --steps 96 --smoke --output validation/vo_smoke_NEW --device cpu --eval-every 0
```

## 云端训练与可视化

以下为同步新源码后手动运行的命令，正式训练状态以本轮验收result.json及运行manifest为准。先核对依赖和实际CUDA前后向，再使用全新输出目录；已有目录拒绝覆盖。

```bash
python -B -m scripts.cloud_preflight --config configs/fast.yaml --device cuda
python -u scripts/train.py --config configs/fast.yaml --experiment Full --seed 0 --episodes 3000 --runs-root outputs/sustained_encirclement_v5 --device cuda
```

五组只替换实验名 `Base w_o_I1 w_o_I2 w_o_I3 Full`。3000回合包含热身，早失败会使交互步数不同；严格学习对照可统一 `--steps` 和多个独立种子。是否并行需云机实测，不把本机CPU或旧4060测量当作5060 Ti吞吐结论。直接训练入口仍预置Numba3、OMP/MKL/OpenBLAS各1的进程线程预算；VO对至少256个receiver任务使用最多3线程并行，小批量环境执行使用串行内核。

本轮不启动正式长训。最终冻结源码全套测试769项通过（116.21 s）；物理验收见 `validation/sustained_v5_physical_frozen/result.json`。CPU/CUDA短训练分别位于 `validation/sustained_v5_frozen_smoke_cpu` 与 `validation/sustained_v5_frozen_smoke_cuda`，各96环境步、83次实际更新；烟雾模式使用batch8、网络32×32及7步回报，仅检查真实Replay、梯度与参数更新链路，不作为正式配置的速度或收敛证据。标准出生点局部信息规则诊断与人工初始围住的物理可达性诊断分别记账，后者不代表模型已学会围捕。

```powershell
python -B -m scripts.visualize --speed 1 --hold-end
```

默认读取配置中的 `outputs/sustained_encirclement_v5`；preview/rules不需要模型。rules_local遵守合法目标信息与已送达伙伴缓存，rules_ideal仅为明确标注的真值诊断。`--show-withdrawal` 为旧的捕获即终止配置保留独立规则段兼容，当前650步任务不会额外追加第二段。

## 轻量保存与历史

`outputs/sustained_encirclement_v5/<experiment>/seedN/` 保留 latest.pt、实际评估后的best.pt、CSV、resolved_config、run_manifest及gzip诊断。只保存部署Actor、配置和元数据，不保存Replay/Critic/优化器/完整仿真器，`--resume`明确拒绝。兼容的当前版本Actor warm-start也必须新目录，并核对任务/奖励/观测/架构及物理限幅，仅迁移网络权重，优化器与Replay重建。

训练默认不保存每步完整大轨迹；需要时显式 `--train-trace-every 10`，诊断复制CSV需 `--export-diagnostics-csv`。历史outputs、validation、失败轨迹及两个清理manifest均保留。迁移前README、HANDOFF、handoff_state和契约完整归档于 [历史v4文档](docs/history/pre_vo_v4/HISTORY.md)，旧源码、配置、测试保存在本次验证的 `pre_vo_source.zip`。历史预测I1、GRU及旧奖励的计时/成功结果只属于各自版本，不与VO v5混图。

2026-11-07项目瘦身后，低频使用的191个历史轨迹、诊断和源码文件已逐字校验后压缩归档，原始路径见 `validation/archive_index.json`。模型、outputs、有效配置、测试和当前验收摘要仍直接保留。查看与按需恢复方法见 [验证数据目录](validation/README.md)，清理明细见 [瘦身报告](validation/project_cleanup_20261107/REPORT.md)。旧阶段的纯编译缓存压缩包已删除；实验数据不受影响。


奖励与任务语义已改变，v4 Replay/Checkpoint不可用于v5续训或warm-start；使用全新输出目录从头训练。先执行上面的验证及短烟雾检查，再执行云端训练命令。本次没有启动正式长训，短诊断不构成收敛或全场景可达性保证。
