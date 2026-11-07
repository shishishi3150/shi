# 异构 USV–AUV 岛礁协同围捕与阻截

在原工程中精准替换任务：**2 USV + 3 AUV、五个分散决策、完整 10 维力/力矩联合动作 [5,2]**。保留原 3-DOF 欠驱动动力学、MATD3 入口、绘图风格和产出目录。红船由独立脚本控制。学习观测和奖励不使用槽位、参考点、预设搜索航线或指定三角阵型。

**最新动力学约束：全艇禁止倒车。** USV、AUV、红船和I1预测均采用纵向速度下限0；负推力仅用于制动，不能倒行。终局安全完成奖励已按用户要求改为+2；首次围捕+2、过程每步0至0.02不变。

**2026-11-07：I1几何检查继续压缩，I3回报尺度已校准。** I1保留完整20步预测、最多200步制动和每步5个物理子步，用保守轨迹下界减少安全路径的几何精查，并复用于I3布尔安全筛选。I3 `return_scale`由1.0改为0.05，质量门槛仍0.2；同批历史由0个变为324个通过候选。相同I3已启用负载下，本机Full更新约67.56→61.33ms（-9.2%），参数对照一致；相对旧I3零增强运行，恢复I3计算后整体仍增加约23%更新时间，不能混作提速结论。118项相关测试通过，MATD3基础设置不变。详见[I1/I3本次报告](validation/i1_i3_efficiency_20261107/REPORT.md)。云端需同步后用新目录启动。

新对话接手请先读[完整项目交接](docs/HANDOFF.md)，包含用户决定、当前代码/证据、历史隔离、待办和可直接粘贴的开场说明。

**2026-11-07当前主干统一改为单帧MLP，GRU及8帧历史已从所有实验组移除。** Actor只读当前91维本地帧（90个数值特征及1个帧标记，标记不进网络），保留两层256节点MLP；Critic使用当前99维全局状态及完整10维联合动作。Base关闭I1/I2/I3，Full和各消融共用相同主干。保留同类艇参数共享、7步回报及共用局部滤波，因此应称“参数共享、n-step前馈MATD3”，不冒称逐字复现原论文。具体审计、计时和验证见[单帧主干报告](validation/clean_mlp_20261107/REPORT.md)。旧GRU策略需使用旧源码，不转换或续训；旧结果保留，不能与新架构混图。

**当前奖励以[护航向围捕迁移设计报告](validation/reward_design_20261107/REPORT.md)为准。** 默认 `cooperative_positive_v2` 在正奖励中加入AUV速度/航向引导和USV阻截几何，保留失败当步0、首次围捕+2、安全完成+2。本次仅改变上述网络主干；batch256、每环境步一次Critic更新、Actor延迟2、gamma0.99、学习率、网络隐藏层宽度、物理和任务门槛保持原值。默认训练与visualize仍读写 `outputs/positive_reward`；不同架构或奖励版本使用新目录或新种子，非空目录拒绝覆盖。规则成功和短测通过不代表RL已经收敛。上一轮[耗时与输出体积调查](validation/positive_reward_20261107/REPORT.md)保留溯源。

**以下为去除GRU之前的实现优化记录，见[配对验证报告](validation/runtime_optimization_20261107/REPORT.md)。** 当时仅优化单次更新的无参数历史预处理、诊断标量读取、I1积分缓冲和I3空路径/稳定排序，未改当时网络结构。本机4060 Laptop两轮Full更新耗时减少10.95% / 8.77%，Base为2.52% / -0.20%；这些数值只属于当时GRU版本。本次单帧架构以最新报告为准，不能把两轮收益相加。云端任务未被本地修改自动更新。

训练入口仍默认选用 `configs/fast.yaml` 的I3 **256/4**，直接启动时在导入数值库前设置Numba3线程、OMP/MKL/OpenBLAS各1线程。无需额外export或配置参数。一张5060 Ti同时跑五组会共享GPU计算和CPU预测资源；先用启动日志的 `actual_device` 及 `diagnostics/runtime.jsonl.gz` 测单组开销，再决定并行度。以下Linux命令逐组执行，每组3000个完整回合；串行是否提高五组总吞吐仍需云机实测：

~~~bash
mkdir -p launch_logs
for method in Base w_o_I1 w_o_I2 w_o_I3 Full; do
  python -u scripts/train.py --experiment "$method" --seed 0 --episodes 3000 --device cuda > "launch_logs/${method}_seed0.log" 2>&1 || break
done
~~~

`Base / Full / w_o_I1 / w_o_I2 / w_o_I3`统一配置，仅替换experiment。保留batch256、热身10000步、Critic每环境步一更、Actor延迟2、原物理和成功门。默认关闭完整训练轨迹、每1000步记录分项诊断；回合汇总和CSV曲线仍保留，所有诊断采用gzip，不默认复制成大CSV。需要轨迹时显式用 `--train-trace-every 10`，需要重复CSV时用 `--export-diagnostics-csv`。五组各3条Numba线程是线程预算，不绑定CPU。3000回合包含热身，早失败会使累计环境步不同，正式对照建议统一 `--steps`。云机先运行 `python -B -m scripts.cloud_preflight --config configs/fast.yaml --device cuda` 核对实际CUDA前后向；显卡配置与 `device=auto` 本身不能证明昨晚确实用了GPU。

~~~bash
python -B -m scripts.visualize --speed 1 --hold-end
~~~

**最终目标是三艘AUV完成围捕，USV阻截为其争取时间。** 三AUV均距红船2–5 m，最大方位空隙≤160°，红船位于实际包围内部且距各边至少0.25 m，三艇本地跟踪有效，连续保持3 s即确认首次围捕。当前默认随后立即触发红船撤退，五艇继续当前策略和训练，直到本局累计650步（97.5 s）结束；不是成功后再加650步。未围捕仍保留900步（135 s）超时失败，首次围捕晚于650步则当步结束；接触、突破等真实失败始终可优先提前终止。USV阻截仍为辅助，不是成功的附加硬门。当前证据见[650步后段训练验收](validation/post_capture_650_20261106/REPORT.md)，此前独立规则演示是历史记录。

当前已实现 Base、I1 预测安全动作筛选、I2 年龄与不确定性补偿、I3 训练期历史完整联合动作增强。机制、公式、信息权限及局限见 [任务契约](docs/task_contract.md)。[五艇初次验收报告](validation/five_agent_acceptance/REPORT.md)保留工程历史，[旧报告](validation/REPORT.md)属历史六艇版本；这些数值不能代表当前围捕目标下的训练效果。

I1计算优化和备用制动检查的后续验证见 [安全与耗时复核](validation/five_agent_acceptance/i1_optimization/REPORT.md)。该验证保留环境步长、积分子步、奖励和轻量策略保存规则；不是长训成功率结论。

两艘USV从C岛礁右下外侧的(38,22)、(41.5,22)同侧前出，三座岛礁中心间距仍为原来的1.1倍。红船从(11,30)起步，使用**APF加局部切向绕行**：保留保护区吸引与岛礁、USV、边界排斥，遇前方USV时选侧绕行；没有中间航点。AUV1保留(12,36)，AUV2位于B、C下方的(29,13)，AUV3位于C右上方的(40,36)，三艇初速度零。规则USV收到合法目标估计后立即阻截。本轮规则seed5/3仍在48.15/47.85秒首次围捕，均继续到97.5秒；这不代表强化学习已学会。后段复用当前env.step、原物理、传感通信、过程奖励与Replay，Actor不会被规则控制器接管。整艇越过入口仅作诊断，不提前终止；实时严格包围可能失效，不把首次capture当成全程围住。入侵shadow在capture后停用，不能把脚本撤退算为USV因果贡献。地图、绘图风格和当前有效几何连线均保留。

## 检查与场景验收

在项目根目录运行；先使用现有 Python 环境，缺依赖时再安装 requirements。

~~~powershell
python -m pip install -r requirements.txt
python check_config.py --experiment Full --device cpu
python -B -m pytest -q -p no:cacheprovider
python -m scripts.validate_scene --experiment Base --red-episodes 100 --rule-episodes 30 --ideal-episodes 3 --base-seed 0 --output validation/five_agent_acceptance/scene_example
~~~

rules_local 与学习策略受相同信息权限约束；rules_ideal 是显式标注的真值诊断。场景输出包括原风格地图、实际轨迹、关键事件、跟踪与通信曲线。只统计实际运行的回合，非空输出目录拒绝覆盖。

## 八个核心配置与 I3 消融

| 配置 | I1 | I2 | I3 |
| --- | --- | --- | --- |
| Base | 关 | 关 | 关 |
| Full | 开 | 开 | 开 |
| w_o_I1 | 关 | 开 | 开 |
| w_o_I2 | 开 | 关 | 开 |
| w_o_I3 | 开 | 开 | 关 |
| I1_only | 开 | 关 | 关 |
| I2_only | 关 | 开 | 关 |
| I3_only | 关 | 关 | 开 |

另外五个实际预设是 Traditional_CV、Random_History、No_State_Matching、No_Safety_Screening、Hard_Max，共 **13 个**。Traditional_CV 使用传统接收端常速度预测；四个历史变体仅替换 Full 的对应 I3 机制。Base 仍使用共用传感器和发送端局部滤波，但接收端保持原观测时刻 track；公平性细节见契约。

先做实现烟雾检查：

~~~powershell
python -m scripts.train --experiment Base --seed 1100 --steps 96 --smoke --output validation/five_agent_acceptance/smoke_example --device cpu --eval-every 0
~~~

smoke 缩小网络、Replay、batch 和 warmup，并默认在32步外部截断，只用于实现检查。短学习预实验使用 pilot：batch32、warmup64、Replay2000，场景、动力学和奖励保持一致。以下覆盖八个核心方法，使用同种子、同交互预算：

~~~powershell
$coreMethods = @("Base", "Full", "w_o_I1", "w_o_I2", "w_o_I3", "I1_only", "I2_only", "I3_only")
foreach ($method in $coreMethods) {
    python -m scripts.train --config configs/pilot.yaml --experiment $method --seed 1100 --steps 600 --runs-root outputs --device cpu --eval-every 0
}
~~~

I3 对照与传统预测基线：

~~~powershell
$historyMethods = @("Random_History", "No_State_Matching", "No_Safety_Screening", "Hard_Max")
foreach ($method in $historyMethods) {
    python -m scripts.train --config configs/pilot.yaml --experiment $method --seed 1100 --steps 600 --runs-root outputs --device cpu --eval-every 0
}
python -m scripts.train --config configs/pilot.yaml --experiment Traditional_CV --seed 1100 --steps 600 --runs-root outputs --device cpu --eval-every 0
~~~

选择未使用的目录或种子；不要覆盖历史 outputs/Base/seed0。600步可以检查更新与日志，不能证明已学会阻截。当前 pilot 的源码版本、实际预算及结果以五艇验收报告为准。

当前默认奖励为 `cooperative_positive_v2`：每艇每步按任务质量赚取0至0.02分，首次围捕一次性+2，安全完成+2；碰撞、触礁、越界、突破、超时当步均为0，没有时间扣分或生存常数奖。AUV奖励接近自由围捕距离带、形成包围、跟踪和保持，新增带外闭合/带内跟随目标的速度与航向质量，穿礁方向不付运动引导奖；USV奖励接近、保护区方向的阻截几何、有效同状态阻截、支持与正贡献，撤退阶段用持续追围支持替代入侵信用。没有指定槽位、旋转相位或中间航点。安全净距及动作平滑程度只调低任务收益，不单独发奖。按用户最新要求，安全完成终局奖为+2，每艇整局未折扣宽界为[0,22]；不再保证成功局回报高于所有失败局的累计过程收益。`gamma=0.99`不变。best选择依次比较成功率、较低安全事故率、首次围捕率和折扣回报；所有实验使用相同奖励，不按I1开关加分。权重见[任务契约](docs/task_contract.md)第7节。v1、v2及历史PBRS曲线不能混图。

短训入口（先选全新目录，不覆盖本轮已跑结果）：

~~~powershell
python -m scripts.train --experiment Base --seed 4700 --steps 10320 --runs-root validation/positive_reward_check --device cuda --eval-every 0
python -m scripts.train --experiment Full --seed 4700 --steps 10320 --runs-root validation/positive_reward_check --device cuda --eval-every 0
~~~

10320步使用默认10000步热身和320次梯度更新，仅检查真实回合、有限数值和日志，不证明收敛。非smoke训练预算若不超过warmup会明确拒绝，避免全程随机采集。正式实验使用相同场景、奖励、网络、独立种子和交互预算；默认I3搜索预算仍为256/4。各组均应保留冻结评估：

~~~powershell
python -m scripts.train --experiment Base --seed 2100 --steps 1000000 --runs-root outputs/positive_reward --device cuda --eval-every 100000 --eval-episodes 30
python -m scripts.train --experiment Full --seed 2100 --steps 1000000 --runs-root outputs/positive_reward --device cuda --eval-every 100000 --eval-episodes 30
~~~

默认 batch256、warmup10000、Replay20000、n-step7、gamma0.99。其余核心与 I3 对照共享 gamma、奖励、网络、场景和预算；不应只为 Full 选择训练配置或种子。

## 冻结评估、回放与曲线

~~~powershell
python -m scripts.evaluate --checkpoint outputs/Base/seed1100/latest.pt --episodes 30 --base-seed 2000000 --variant fixed --suite nominal --output validation/five_agent_acceptance/base_eval_example.csv
python -m scripts.evaluate --checkpoint outputs/Base/seed1100/latest.pt --episodes 30 --base-seed 2000000 --variant fixed --suite communication --output validation/five_agent_acceptance/base_comm_example.csv
python -m scripts.evaluate --checkpoint outputs/Base/seed1100/latest.pt --episodes 30 --base-seed 2000000 --variant fixed --suite collaboration --output validation/five_agent_acceptance/base_collaboration_example.csv
python -m scripts.visualize --checkpoint outputs/Base/seed1100/latest.pt --variant fixed
python -m scripts.visualize --preview --experiment Base
python -B -m scripts.visualize --rules --experiment Full --seed 5 --speed 1 --show-withdrawal --hold-end
python -m scripts.quxian --runs-root outputs --experiments Base Full --seeds 1100 --window-episodes 10 --output-dir validation/five_agent_acceptance/curves_example
~~~

还提供 screen 套件。通信套件覆盖0/3/10/20/40步延迟和0/5/10/20/30%丢包。协作套件包括只保留 USV、只保留 AUV、分别移除 USV/AUV、关闭伙伴状态、延迟及乱序伙伴状态；伙伴状态消融仍保留目标观测通信。preview 给蓝方零执行器命令；rules 使用合法局部信息的规则控制器，无需模型。当前默认不加撤离选项也会在首次围捕后继续同一回合到累计650步；`--show-withdrawal` 只为旧配置（缺少 `post_capture_end_step` 或为0）保留附加独立规则段兼容。`--hold-end` 在结束后停留；可选 `--start-paused`、`--speed 2`，Esc退出。曲线只用已结束回合，跨种子方差带需要多个独立种子。

## 轻量保存与历史兼容

~~~text
outputs/positive_reward/<experiment>/seedN/
  latest.pt
  best.pt
  train.csv
  updates.csv
  resolved_config.yaml
  run_manifest.json
  diagnostics/
~~~

latest/best **只保存部署所需 Actor、配置和元数据**；latest 覆盖写，best 仅在实际评估后产生。保留 CSV、压缩轨迹及诊断；不保存 Replay、优化器、Critic 或完整仿真器，不累积逐步 checkpoint。Replay 仅占进程内存，check_config 会估计开销。

完整状态 --resume 明确拒绝。Actor warm start 必须建立新目录：

~~~powershell
python -m scripts.train --config configs/pilot.yaml --experiment Base --seed 1101 --steps 600 --runs-root outputs --warm-start outputs/Base/seed1100/latest.pt --device cpu
~~~

Critic、Replay、优化器与计数器重新初始化，manifest 记录来源 SHA 和训练步数。观测 schema、艇数或网络不兼容的历史权重明确拒绝，原文件及历史结果保留。清理仅针对确认无用的代码，不把历史实验产出当垃圾删除。
