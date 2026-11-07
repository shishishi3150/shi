> **2026-11-07 最新：I1几何压缩 + I3回报尺度修正。** 完整预测/制动时域与物理子步未缩短，增加保守径向轨迹证书，并在I3布尔安全筛选复用；I3 `return_scale` 1.0→0.05，匹配实际约0.03–0.05的7步回报，质量门槛0.2不变。冻结真实Replay实测I1批量投影11.56→9.61ms（-16.9%），相同I3已启用条件下Full更新67.56→61.33ms（-9.2%），两组52次更新后网络参数差0。相同检索池从0到324个通过候选，48次更新产生3658条非零增强。相对旧I3零增强的运行，当前Full更新49.79→61.33ms，净增加23.2%；不要把恢复功能的开销与同功能实现优化混淆。两项先算Q/逐轮候选筛选试验实测更慢，已撤回生产；`matd3/agent.py`逐字节恢复本轮前版本，原筛选与日志字段不变。118项相关测试通过；MATD3基础参数、终局+2、禁止倒车均保持。未跑新长期环境训练，未同步云端。报告 `validation/i1_i3_efficiency_20261107/REPORT.md`，当前source hash `e08bd5bbd457d89c5b6d121e6ddc9b0f6a155f8df8376d44b93b1f8d9052eb0d`。

> **2026-11-07 上一次要求：动力学禁止全部倒车。** 共用ship.allow_reverse=false，USV/AUV/红船及I1对象/编译预测的纵向速度下限均为0；原积分器每子步在位置更新前执行截断。负推力仍作制动，不禁左右转向或横荡。规则艇不再请求负纵向速度，当前配置校验拒绝重新开启倒车。旧允许倒车任务/策略需原源码，当前使用新目录训练。奖励仍为终局+2、首次围捕+2、过程0至0.02。该次修改按用户要求未跑测试；本轮相关测试另见顶部最新记录。task hash `1e568630b80677857ba801542e57730c1ca372d315cfe42aadc90cf13d23b83d`，当时source hash `0f0c277d481c1734b766fec8bae18790c267277cb92eab7cf847e98265c982ca`。

> **2026-11-07 用户最新调整：安全完成终局奖20→2；首次围捕+2、普通过程每步0至0.02不变。** 已取消要求终局奖大于19.98的旧校验，保留其余正值/权重/总预算校验。因此不再保证成功局累计回报高于所有失败局；基础训练设置和单帧MLP不变。按用户明确要求，本次未运行测试；相关旧期望值已同步更新。下方报告中终局20及旧奖励hash为历史。当前奖励hash `d24a75e2ad2cc9e67a87db71a7ee2a8a9452b0f445b732a321ef64ea1a67f984`。

> **2026-11-07 第五轮：按用户授权统一移除GRU和8帧历史，当前状态优先。** 所有13种预设统一为当前单帧MLP；Actor输入91维帧/90个有效数值特征，Critic为99维当前中央状态+10维完整联合动作。无GRU/RNN/LSTM；HistoryEnv只在生产中保留1帧。I3训练期历史动作检索保留，不能误删。旧GRU权重需旧源码，新版本必须新建训练目录，不能混图。
>
> 基线应命名“参数共享、n-step前馈MATD3”。保留原7步回报、角色共享、多头Critic、局部滤波、巡航初始化/探索、Huber和裁剪，均为公开共用设置；Base的I1/I2/I3全关。没有通过削减Base合法信息、训练预算或任务门槛保证排名。batch256、热身10000、Critic每环境步一次、Actor延迟2、学习率、奖励及安全预算均不变。
>
> 本机真实冻结Replay配对：Base44.25→12.55ms/update（-71.63%），Full77.42→45.89ms（-40.72%）。这是架构变更，非数值等价或质量验收。I2五艇接收端增量约0.09ms（1来源）、0.55ms（3来源）、0.69ms（4来源），不新增神经网络更新分支。545项回归通过；Base/Full各真实10128环境步、保留10000热身，之后128更新，CUDA与实际策略加载通过。未跑长期成功率对照，未改云端。
>
> 当前源码hash `74f004d802673ccbae0148a5346cb58e098b6464e7807f0aca48bcb8858702dd`，奖励hash不变；51个生产/配置文件中13个变化（6个Python+7个配置），清单、证据与边界见 `validation/clean_mlp_20261107/REPORT.md` 和 `evidence.json`。最新机器记录键为 `current_update_20261107_clean_mlp_backbone`。下方“GRU尚未删除/生产未改/旧hash”仅代表此前审计阶段。

> **2026-11-07 第四轮：GRU底座与耗时审计；本轮生产代码和配置未改。** 当前Base实际上强制带GRU8，三份奖励重做前的源码备份也已存在，不是新I1/I2/I3之一。每次更新平均13次Actor/Critic历史编码，不能把它称为普通单帧MATD3的必然成本。当前51个生产Python/YAML与上一轮结束快照全部一致。
>
> 本机当前代码同步分段：Base更新45.902 ms，Full85.472 ms；其中GRU前向/预处理15.080/19.663 ms，反向合计15.400/20.017 ms，Full的I1训练投影21.010 ms。完整独占账与128步真实反馈见 `validation/runtime_attribution_20261107/REPORT.md`。计时插桩与原更新训练状态精确相等。
>
> 单独计算消融：相同采样及batch下，当前Base43.282 ms，诊断克隆旁路GRU后17.286 ms（约减少60.06%）。该克隆将hidden置零并去掉GRU梯度，不是正式MLP替代或质量验收；未删正式GRU、未改模型、未保存Replay。不要把此数字写成保质量的正式训练提速。
>
> I2风险：Base已接收位置/速度/年龄/方差/置信度及8帧历史，存在学习部分补偿的条件；历史跨度1.05s大于默认通信延迟0.45s。I2仍有显式状态/协方差传播及保守多源融合，Base接收端hold未偷偷调用它。实际削弱幅度未完成学习消融；单帧MLP也可能利用速度和年龄，删GRU不保证I2更优。建议统一主干并做MLP/GRU8×I2关/开的四组对照。本轮用户只要求核查，没有将GRU删除纳入正式实现。最新机器记录为 `current_update_20261107_architecture_runtime_audit`。

> **2026-11-07 第三轮：训练实现提速，当前状态优先于以下阶段记录。** 用户要求继续压缩Base及创新模块的计算时间，同时保留基础学习设置和训练质量。本轮已修改 `matd3/agent.py`、`matd3/networks.py`、`matd3/history_enhancement.py`、`usv_env/safety_kernel.py`、`usv_env/safety_batch.py`，分别优化单次update的无参数历史预处理、诊断读取、I3空路径/稳定排序和I1积分缓冲。奖励仍为 `cooperative_positive_v2`；所有YAML、更新频率、batch、warmup、网络结构、精度、动力学、任务门槛及I1/I3预算不变。
>
> 全量526项通过。两轮五组的所测参数、梯度、Adam状态及采样精确相等，各组64步新旧环境反馈相同；既有Actor-only模型48个观测的动作输出相同。这是有限等价性验证，不是长期收敛证明。51个生产源码/配置文件逐一比较，只有上述5个实现文件变化，完整组装配置相同。
>
> 本机4060 Laptop两轮Full更新耗时减少10.95% / 8.77%；Base为2.52% / -0.20%，不能声称基线稳定显著提速。其余三组观测减少约5%～11%，第二轮部分组CV较高；不同Replay规模/分布及运行状态未被分离，跨轮绝对耗时差异的原因尚未确定。I1/I3非空压力样例与完整数值对照见 `validation/runtime_optimization_20261107/REPORT.md`，机器证据为同目录 `final_evidence.json` 和 `source_integrity.json`。
>
> 当前源码hash为 `0fcb344c9024f846569e6616c8442607099db14cd7e55e7e680bc2f985be822b`，奖励hash仍为 `c9979e697a279681f4ba46a231e9e1b111096880be5cf3e9afeb3dceaf785511`。需一并同步5个文件后由新启动的云端进程使用；本轮未连接或操作云端，也未启动正式长训。更新耗时收益不能直接视为五组并发总时长收益。
>
> 以下奖励阶段“整个matd3未动/优化已撤回”等描述只对其当时起点和阶段成立，不代表当前源码。保留历史不覆盖；最新机器状态为 `handoff_state.json` 的 `current_update_20261107_runtime_optimization`。

> **2026-11-07 第二轮：护航向围捕的奖励迁移设计，优先于下文v1记录。** 当前默认 `cooperative_positive_v2`，本轮重新逐行核对用户朋友的两个护航附件，补充了AUV距离门控的速度/航向引导和USV保护区方向的阻截几何，保留自由围捕距离带及原成功标准。未照搬槽位、旋转相位、USV跟随红船速度或原函数有跳变的两段惩罚。设计目标、逐项迁移表、公式和验证顺序见 `validation/reward_design_20261107/DESIGN.md`；证据和局限见同目录REPORT。
>
> 本轮验收结果：全量515项通过、10条规则/失败物理审计、v1与最终v2各14000环境步/4000次更新。共同3场冻结开发评估均0/3成功且保持0 s；v2末端AUV距目标平均19.923 m，v1为36.023 m，但v2折扣回报更低。只能说本次短训接近改善，不能说已学会围捕或收敛。详细表见本轮REPORT。
>
> 用户“不改基础设置”的约束继续有效。本轮整个 `matd3/`、`scripts/train.py`、除reward.py以外的所有usv_env文件及所有预设逐字节不变，模板只有reward段变化。本轮不再调整日志、并行度、更新频率、batch、学习率、gamma、热身、网络或物理。新旧奖励不可直接比较原生分数，配对冻结策略使用相同v2评估奖励并同时看真实成功/安全/围捕指标。
>
> 以本轮 `foundation_integrity.json`、`full_tests_final.xml`、`physical_audit_final/`、`train_v2_verified/`、`paired_frozen_comparison.json` 为准。初始未做距离门控的 `train_v2/` 与 `physical_audit/` 是探索版本；不要把其结果冒充最终候选。v1兼容保留，不混入v2训练曲线；本轮没有启动3000回合正式训练，不声称长期收敛或Full必然第一。机器可读最新状态见 `handoff_state.json` 的 `current_update_20261107_reward_design`。

> **2026-11-07 当前状态，优先于下文所有历史记录：** 用户最新明确要求“不改更新频率，基础的这些先不要改”。本轮所有MATD3计算优化已撤回，`matd3/`及配置预设与本轮修改前备份逐字节一致；`scene_template.yaml`仅reward段改变。batch256、热身10000、Critic每环境步一次、Actor延迟2、gamma0.99、学习率、网络、物理、任务门槛和650步成功后段均保持原值。
>
> 默认奖励现为 `cooperative_positive_v1`：每艇过程收益0至0.02，首次capture一次性+2，安全完成+20，所有失败终止当步所有奖励为0；无时间扣分或存活常数奖。标准五艇下失败未折扣回报上界19.98小于成功奖20；这不保证任意折扣轨迹全局排序或Full永远第一。所有消融同奖励，best按成功率、安全事故、首次capture和折扣回报选择。护航附件仅作连续质量参考；AUV/USV任务权重见task_contract第7节。
>
> 新默认输出为 `outputs/positive_reward`，历史结果原样保留。全训练轨迹默认关闭、分项诊断每1000次更新写一次、全部诊断gzip、默认不额外导出重复CSV。**这些是日志采样，MATD3梯度更新频率未改。** 启动打印实际设备/GPU，runtime记录各阶段耗时，manifest定期落盘。旧80 MB主要为训练轨迹和奖励分项，不是大模型。云端单张5060 Ti同时五组共享GPU/CPU；旧记录未能确认实际CUDA使用，不能断言全部耗时原因或声称本轮已显著提速。
>
> 当前证据以 `validation/positive_reward_20261107/REPORT.md` 为准；`foundation_integrity.json`记录基础完整性，`physical_audit_final/`为完整物理审计，`default_training/`为保留10000步热身的Base/Full短测。曾撤回的优化探针及不完整 `physical_audit/` 仅供溯源，不能作为当前提速或验收结论。长期3000回合正式训练未由本轮启动；规则成功/有限更新不等于RL收敛。机器可读最新状态见 `handoff_state.json` 的 `current_update_20261107`，其余旧resolved config与hash为历史。

> **下文自2026-11-06起为历史记录；与以上最新状态冲突时以上文为准。**

> **2026-11-06 当前训练语义更新（优先于下文历史记录）：** 用户已授权“成功之后继续训练到650步结束”。默认 `task.post_capture_end_step=650` 是从reset起累计步数，即97.5 s，不是成功后再加650步。原AUV围捕标准不改；首次确认后锁存capture和时间、立即触发红船WITHDRAW，不终止、不切换蓝方控制器，过程奖励/传感通信/I1/Replay/学习更新继续。正常到650步以success结束；未capture保留900步timeout；晚于650首次capture则当步结束；安全失败仍可提前结束并保留capture记录。首次capture不要求以后每步持续满足严格包围，实时门槛及成本仍逐步记录。当前原规则seed3/5首次capture仍为319/321步，均真实运行到650步；这不是学成策略证据。新验收目录 `validation/post_capture_650_20261106/`，其中REPORT与manifest记录本次实测和hash。
>
> 入侵shadow在capture后冻结/隐藏、旧audit清除并停止后续阻截审计，gap/progress归零，避免把自主撤退归为USV信用；USV既有阻截成本不再因假gap降低，Shapley保持中性。当前配置不再额外追加独立规则撤离，Actor回放始终是Actor。旧配置缺字段或显式为0才保留立即终止和旧 `--show-withdrawal` 独立规则段兼容。本文件下文独立后段、120 s撤离上限及原650之前终止口径均属于历史；本轮不修改地图、动力学、网络/观测维度、奖励权重、gamma、入口、输出目录或latest/best轻量保存，也未启动正式训练。
>
> 奖励终检更新：当前默认为 `cooperative_cost_v2`，下文旧奖励hash/缩放记录属于历史。每步过程成本最多0.1、**正常回合真正结束的最后一步**为0、所有失败-10；首次capture和其后非终止步继续原过程成本。900步整体上界预算仍为[-100,0]；详见 `validation/reward_final_audit_20261106/REPORT.md` 和task_contract第7节。train.csv含折扣回报以及首次capture/后段步数/整艇退离诊断；不要把规则成功或短训更新通过写成已收敛。

# 项目交接：异构 USV–AUV 岛礁协同围捕、阻截与撤离演示

更新时间：2026-11-06，Asia/Shanghai。项目根目录：`C:\Users\36043\Desktop\shi`。

本文面向接手本工程的新对话，记录用户决定、现有实现、实际证据、历史问题及建议的后续顺序。本文是状态交接，不表示启动了新的训练，也不表示用户已要求更改撤退触发机制。机器可读快照见同目录 `handoff_state.json`；公式与详细接口以 `docs/task_contract.md` 和当前代码为准。

后续清理记录（2026-11-05）：用户明确要求删除无用验证文件后，已删除scene_behavior早期规则调参的145个中间CSV，共329.34 MiB；报告、图片、最终验收、当前回放、I1优化、奖励审计和训练输出保留。部分历史JSON中指向已删中间CSV的路径不再可用，不能声称这些旧轨迹仍完整保留。逐文件清单见 `docs/validation_cleanup_manifest.json`，解释见 `validation/five_agent_acceptance/scene_behavior/CLEANUP.md`。生产代码与交接配置未变。

## 1. 接手先读这部分

这是用户迭代很久的成熟科研工程。用户要求在原工程精准替换场景和创新模块，保留架构、入口、输出位置、图形风格与轻量策略保存，不接受重新搭一套工程或叠加V3/V4.3之类运行层。

当前是 **2艘蓝方USV + 3艘蓝方AUV，对抗1艘脚本红船**。五个蓝方智能体各自决策，每艇两个物理执行器，联合动作 `[5,2]`，共10维。

用户最终明确的主任务：**USV通过侧面阻截和干扰推进，为AUV争取时间；三艘AUV形成有效围捕即任务成功。** 仅守住保护区到时限、仅USV挡住红船、仅持续探测，都不算成功。

当前训练和演示是同一回合：首次AUV围捕成功触发红船撤退，蓝方继续用当前策略追堵、追围并参与奖励和Replay，直到累计650步；旧的终局后独立规则段仅为历史配置兼容保留。首次围捕判据不变，未成功仍保留900步上限。

**当前看到的成功画面来自规则控制器，不是强化学习已经学会。** 当前地图下尚未完成统一多种子正式训练、完整消融或泛化验收。

三个算法模块仍是：I1真实动力学预测与备用制动安全筛选；I2旧消息的年龄与不确定性补偿；I3历史完整联合动作的保守训练价值增强。代码接通与创新有效要分开看。I1不能称严格保证的CBF；I3历史真实短训中合格候选和实际增强仍为零。

接手时无需从头重构或重新追调规则成功率。用户已明确：规则只负责探路、看场景是否合理可达，不是项目重点，不再要求规则100%成功。

## 2. 最近对话停在什么地方

最近几轮用户与助手的问答，按当前有效口径记录：

1. 用户希望红船被挡后会绕行、入侵失败后逃走，不再停住或死冲保护区。允许在确有需要时小幅移动红船或C附近USV，但本轮最终没有再改出生位置。
2. 用户此前选择“保留围捕成功，另验证撤离过程”；2026-11-06最新授权已改为首次capture触发撤退，继续同一训练回合至累计650步，未将整艇退离或始终严格围住改为新的硬成功门。
3. 第一版后续演示让蓝方制动停止，用户随后明确纠正：**“红船跑走也要继续围捕和驱逐”**。现已改为五艇继续用本地估计行动，保留传感通信与Full的I1。
4. 用户问当前创新点是什么，已解释为I1/I2/I3；红船绕行、撤退和蓝方追围属于场景行为，不冒充新的算法创新。
5. 用户追问：**“红船撤退的触发逻辑是什么？蓝船没有阻挡，它还会撤退吗？”**
6. 已核对代码并明确答复：撤退由AUV围捕成功触发，与USV是否实际有贡献没有直接硬绑定。没有USV但AUV仍围捕成功，也会撤退；没有围捕成功则不触发。它目前不是红船根据自身感知自主判断“冲不过去”而逃走。
7. 用户接着要求全面交接，准备开新对话。本轮交接不擅自把上一个问题理解为已经授权改成“USV压力触发撤退”。

新对话可从“当前触发机制是否符合用户想要的叙事”继续讨论；如果用户仍只问原理，就先解释。如果用户明确要求修改，再据其意图精准落地。不要重复询问已经确认的“训练成功仍为AUV围捕”或“撤退时蓝方继续追围”。

## 3. 用户约束与工作方式

### 3.1 工程与数据

- 只改原工程。保留 `usv_env/`、`matd3/`、`scripts/`、`configs/`、`outputs/`、`validation/` 等结构及原入口。
- 不引入另一个V3/V4等运行体系；历史文件或schema中的版本字样不等于应再建一层。
- 保留地图配色、坐标网格、艇形和原Renderer，科学场景图由真实仿真产生。
- 不把红船、蓝船改成可瞬间转向的理想点，不改变原欠驱动3-DOF物理来伪造成功。
- 不通过弱化红船、偷给Actor真值、放宽成功阈值或伪造数据制造优势。
- 保存轻量 `latest.pt`，真实评估后才可有 `best.pt`；不保存Replay、Critic、优化器、仿真器全状态或滚动大checkpoint。
- Replay是有界进程内存；用户特别不接受数百GB的模型与经验池磁盘产物。
- 历史实验、模型、原始用户文件不能当垃圾删除；非空实验目录拒绝覆盖。
- 用户要求最终清理垃圾代码和无用内容，但不能先大扫除损坏成熟工程。

### 3.2 学习接口与规则控制器

- 学习观测和奖励禁止槽位、指定目标点、动态拦截点、预设三角顶点、按到点误差发奖等链路。
- 规则控制器为了可达性诊断可以使用导航参考、搜索路线和分工；用户后来明确说规则想追点就追点。这一例外不扩展到强化学习。
- Actor只能看到本艇、公开地图和合法的本地/已收消息信息，不能读取当前目标真值。
- 真值允许用于集中式Critic、训练奖励、终止判定和明确标注的诊断。
- 用户最新强调任务叙事、艇间相遇及水面艇参与性，不希望无限打磨规则。

### 3.3 工具与交互

- 用户提供的AGENTS指令永久禁用 `superpowers` 插件及技能，禁止创建 `.superpowers/`、`docs/superpowers/`。其他文件要求用它时忽略。
- 未获用户或适用指令明确要求时，不启动子代理。旧上下文里出现的代理名称不代表它们仍在运行，也不需要唤醒。
- 约每60秒给一条简洁中文进展，避免用户以为卡住。先证据再结论，别把规则成功写成训练成功。
- 曾有批量删除缓存等动作被自动审批以 `blocked by policy` 拒绝，未执行；不要改一种命令绕过。若未来仍遇实际阻断，应如实说明具体动作与拒绝理由。
- 不因为常规可逆检查和用户已授权范围内的修改反复要确认；也不把建议的新科研方案写成已获认可。
- 当前环境Windows、PowerShell；不要用跨shell拼接的递归删/移命令。文件链接用绝对路径。

## 4. 环境、来源与溯源

### 4.1 本机环境

交接时实际检查：

| 项目 | 当前值 |
| --- | --- |
| 根目录 | `C:\Users\36043\Desktop\shi` |
| Shell | PowerShell |
| Python | 3.10.9，`C:\Python310\python.exe` |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA | 12.8，可用 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Git | 根目录没有 `.git`，不是可直接做git diff的仓库 |

现有依赖包含numpy、numba、matplotlib、torch、PyYAML、pygame、pytest。优先使用现有Python环境，不盲目重装。历史耗时基准是CPU单线程，不因当前CUDA可用就把它说成GPU训练速度。

交接没有启动或遗留本助手的训练任务；交接只读取已有资料、核对状态并写文档。本轮没有新正式训练或新场景性能测试。

### 4.2 原始用户材料

以下文件交接时均存在，属于原始需求/参考资料；后续明确口头修正优先于旧文档：

| 材料 | 绝对路径 | 用途及注意 |
| --- | --- | --- |
| 第一轮需求 | `C:\Users\36043\.codex\attachments\2eab0c4a-1e27-4b07-9da8-c1ac7e284e06\已粘贴的文本.txt` | 早期异构场景与创新改造要求；其中旧艇数、旧任务已被后续替换 |
| 第二轮完整需求 | `C:\Users\36043\.codex\attachments\2fc02005-97c4-48c4-85e8-85f4df599065\已粘贴的文本.txt` | 五艇、无槽位、信息公平性、学习/消融与验收基础要求；旧坐标和旧终局按之后用户修正 |
| 旧奖励说明 | `E:\xwechat_files\wxid_a3whuunou15322_99c9\msg\file\2026-10\奖励项与权重说明.md` | 旧FormationEnvironment编队参考奖励，不能直接恢复到当前任务 |
| 旧奖励代码 | `E:\xwechat_files\wxid_a3whuunou15322_99c9\msg\file\2026-10\reward_function.py` | 参考附件，不是当前生产奖励入口 |

用户还曾提供截图 `C:\Users\36043\AppData\Local\Temp\codex-clipboard-e610f5d2-d9db-40a3-92ba-4e918dee2f3f.png` 来讨论红船起步直线问题；该临时图片是否仍存在，接手时再查。不要把附件中的历史说明自动升级成新指令。

### 4.3 当前代码和配置身份

交接实际计算，生产源码hash为：

```text
0305e5c2275a3527d0a536475598f18729e4140dfedf1c346d6f1f52decf7a5c
```

## 21. 2026-11-06 奖励重构接手点

最新：用户随后要求过程项外层系数全部乘2.5（包括发现、安全、动作及偏航项），终局保持缩小10倍后的值；当前reward hash为 `b0571394672b58d8c6ac8291361217273b3f3d659ff8b4512497635c679dc33f`。新尺度验证见 `validation/process_scale_20261106/REPORT.md`，旧训练不代表新尺度。

- 2026-11-06 后续用户要求全部终局奖惩除以10；当前值为成功+3、突破-3、超时-1.5、碰撞/触礁/越界-5、异常-10。过程项不变，新reward hash为 `5ac4d2965d1d0282122c4ba921e81d32a8268fe17eb50ecf568de4a412258776`；旧预验收数值仍对应缩放前配置。
- 缩放前预验收的生产默认 `gamma=0.99`，潜势权重 `tracking=0.18`、`hold=0.22`；reward hash 为 `4fc5d7409ae8f2bfcbf507f8c33b9fd22791a15e80273456a4d9330ae2f967cb`。
- 当前预验收目录为 `validation/reward_redesign_20261106/`，含测试JUnit、13条真实奖励审计、Base/Full 1200步和Base 2400步日志；完整报告为 `validation/reward_redesign_20261106/REPORT.md`。
- 293个全量测试通过；规则成功只用于奖励/场景可达性诊断，当前学习短训0次成功。
- Full真实I3漏斗已记录：36,352 queries，phase/info/quality过滤后 eligible、qualified、selected、nonzero enhancement均为0；禁止为制造非零增强而放宽门槛。
- 下一步若继续训练，应使用同一 gamma、奖励、场景、网络和预算做多种子公平pilot，并保留新输出目录；不要覆盖 `validation/reward_redesign_20261106` 或旧 `outputs`。

`usv_env.config.code_hash()`覆盖 `usv_env/`、`matd3/`、`scripts/` 目录下直接的Python文件。它不覆盖README、tests、validation脚本、YAML；配置身份另由下面的hash描述。只补交接文档不会改变生产源码hash。

| 配置身份 | hash |
| --- | --- |
| task | `582a2197eda4e89b05d6fdace647f53ef016f1de7e572c1898287fa5ac7fcc48` |
| reward | `b0571394672b58d8c6ac8291361217273b3f3d659ff8b4512497635c679dc33f` |
| observation | `86c6e6836b242a8e6a7b0f1b5d74fc79bd75ebd945093f49fdfdb0cf0c176b37` |
| architecture | `c60bd4d3f8be2350e00c06c4d611dd8db5b9eb778cc2598becd2bacd916930fe` |

`handoff_state.json`还记录配置快照、源文件摘要和四份当前回放报告的对应关系，便于新对话检查是否有后续改动。它是交接记录，不是新的配置装配入口。

## 5. 当前场景：以这些坐标为准

海域55×55 m，位置单位米。固定布局没有出生位置/航向扰动；种子仍影响传感、通信与波浪。

| 对象 | 位置 | 航向 / 初始纵荡速度 |
| --- | --- | --- |
| 岛礁A | (19.6,38)，半径3.5 | 静态 |
| 岛礁B | (19.6,16)，半径3.5 | 静态 |
| 岛礁C | (32.8,28.1)，半径3.5 | 静态 |
| 保护区 | (49,27)，半径4 | 静态 |
| 红船 | (11,30) | 向东，0.4 m/s |
| USV1 / 图中B1 | (38,22) | 向西，0 |
| USV2 / 图中B2 | (41.5,22) | 向西，0 |
| AUV1 / 图中Y1 | (12,36) | 向南，0 |
| AUV2 / 图中Y2 | (29,13) | 向北，0 |
| AUV3 / 图中Y3 | (40,36) | 向西，0 |

岛礁A/B/C是地形标签，艇B1/B2是蓝方水面艇，不要混淆。用户要求两USV在同一侧、靠近C或C与保护区之间，体现侧面阻断，不要恢复“一艇守一个出口”的部署。

三岛由原中心(20,37)/(20,17)/(32,28)，绕固定中心(24,27)做1.1倍外扩；岛礁半径未放大。Y2和Y3已按用户要求远离礁缘，避免初学时出生贴礁。USV当前在C东南外侧，不在岛礁内部。

上一轮出生几何记录：USV1/2到最近禁航轮廓间隙约3.67/6.19 m；Y2/Y3约5.77/6.54 m。当前未改位置。它们只表示出生合法，不证明随机行动始终安全。

## 6. 动力学、步长与成功终局

### 6.1 步数必须分清

| 概念 | 当前值 | 含义 |
| --- | --- | --- |
| 环境/策略一步 | 0.15 s | 一次env.step、每艇一次联合动作输入 |
| 内部积分子步 | 0.03 s | 一个环境步用5次积分计算欠驱动动力学与扫掠轨迹，通常保持同一物理动作 |
| 训练最长回合 | 900环境步 = 135 s | 围捕未完成则timeout失败 |
| 已围捕回合终点 | 累计650环境步 = 97.5 s | 首次capture后继续学习；晚于650才capture则当步结束 |
| 围捕连续保持 | 20环境步 = 3 s | 瞬时穿过包围不算成功 |
| 当前seed5规则完成任务 | 321环境步 = 48.15 s | 不含后续撤离 |
| 当前seed3规则完成任务 | 319环境步 = 47.85 s | 不含后续撤离 |
| 独立撤离最多 | 800步 = 120 s | 仅演示段上限，不是训练回合延长 |

删除5个积分子步不会直接让RL回合从900变180，也不会增加策略更新次数；它只减少内部物理计算，可能损害接触检测与运动精度。I1加速已经通过计算实现优化，没删物理子步。尚无当前学成策略的平均完成步数，不能把两个规则样本当作训练完成时间。

### 6.2 原物理形式

```text
M * dν/dt + C(ν)ν + D(ν)ν = τ
ν = [纵荡u, 横荡v, 偏航r]
τ = [纵向力Fu, 0, 偏航力矩Nr]
```

没有直接横向推力。速度参考只是规则/安全算法内部手段，实际进入动力学的是力和力矩。

| 限制/轮廓 | USV与红船 | AUV |
| --- | --- | --- |
| 正纵荡上限 | 0.40 m/s | 0.50 m/s |
| 蓝方倒车下限 | −0.10 m/s | −0.125 m/s |
| 纵向力绝对限幅 | 2.2 N | 3.035 N |
| 偏航力矩绝对限幅 | 1.5 N·m | 1.5 N·m |
| 胶囊轮廓长×宽 | 1.255×0.29 m | 0.60×0.31 m |

横荡绝对上限0.6 m/s、偏航角速度绝对上限0.675 rad/s。纵荡上限与二维实际地速要分开，不能拿0.4当所有地速的严格上限。波浪沿用原规则线性波，波高0–0.04 m；AUV不直接耦合水平波速。

AUV固定潜深3 m，水面龙骨0.5 m，AUV竖向半径0.16 m，竖向安全裕度0.5 m。竖向分离允许AUV和水面艇水平投影交叉；同类艇、USV与红船仍碰撞。礁缘外0.4 m为浅水禁航轮廓；真实接触用swept capsule，不能只看质心距离。这里没有深度控制、分层水深图或AUV实际抓停红船的力。

### 6.3 成功和失败

三艘active AUV同时满足以下条件，连续3 s成功：

1. 每艇距红船中心2–5 m。
2. 围绕红船的最大方位空隙≤160°。
3. 红船在三艇真实位置的凸包内，到各边的最小有符号距离≥0.25 m。
4. 三艘AUV自己的本地track年龄≤1 s，位置误差≤3 m。
5. reset静态通路存在，无接触、突破或数值/场景故障。

几何随真实位置计算，不指定三角顶点、朝向或槽位。安全失败、红船中心扫掠进入保护区先于成功；135 s未完成是timeout失败。USV接近、两艇贡献、shadow差距、红船推力占比放在support_gates，只辅助诊断，不是AUV成功的附加硬门槛。

任务期限属于terminated，bootstrap=0；外部collector截断是truncated，bootstrap=1，必须保留真实最后端点，不能跨reset拼n-step。

## 7. 红船控制和撤退：准确区分现状与设想

### 7.1 推进与绕行

当前红船是 **APF加局部切向绕行**，不再是严格纯APF：

- 保护区吸引1.0、岛礁排斥2.0、USV排斥1.5、边界排斥1.0。
- 这些是各项增益，不是固定“几比几”的实际合力比例；实际大小随净距、方向和叠加变化。
- 正净距内排斥强度为 `min(4, k*(1-max(d,0)/d0)^2)`，范围外为零。
- 礁净距扣除红船外接圆半径0.6275 m和1.25 m余量；基础影响净距6 m。
- `reef_field_x_scale=1.8`额外叠加椭圆等势场梯度，使红船从起步阶段较早弯曲进入B、C之间；实体礁轮廓仍是圆。
- USV中心距5 m内有排斥；边界影响净距2.5 m。
- 前方5 m有USV时，最近前方USV的径向排斥再加0.7倍切向量；左右以公开地图3 m局部探测净距及目标进展选侧，6 m内仍有USV则锁存方向。
- 有阻挡为BYPASS，无前方阻挡为ADVANCE。没有控制航点，不靠计时器强制脱困/撤退，不读取AUV真值切换撤退。
- 常规参考偏航上限0.25×0.675=0.16875 rad/s；合力小或艏向偏差大时会减速。物理执行器限幅未变。

历史策略配置缺 `bypass_gain` 时，RedNavigator默认0，保留旧纯APF行为。诊断停滞计数不是自主放弃入侵逻辑。

### 7.2 历史：独立演示模式什么时候撤退

当前650步同回合模式以文件开头更新和task_contract第5、6节为准；以下是此前独立模式的实现记录，只用于缺少新字段或显式为0的历史配置。

准确调用链：

```text
原env.step依原条件产生success
→ scripts.visualize._run_episode检查show_withdrawal且info.success
→ _run_withdrawal
→ WithdrawalReplay要求原环境已终止且terminal_reason=='success'
→ 复制终局和相关状态
→ navigator.request_withdrawal(exit_goal, reason='capture_confirmed')
→ WITHDRAW锁存，红船向入口外驶离
```

所以：

| 情况 | 当前会不会触发撤退 |
| --- | --- |
| USV阻挡有效且AUV围捕成功，开启演示 | 会 |
| USV没有阻挡/被移除，但AUV仍围捕成功，开启演示 | 也会 |
| USV挡住了，但AUV没围捕成功 | 不会由当前逻辑触发 |
| 红船被突破、接触或超时终止 | 不触发 |
| 正常训练或未开show-withdrawal | 没有这段后续撤离 |

不能描述成“红船自主感知蓝方压力大，所以逃跑”，也不能描述成“USV不阻挡红船就永远不退”。用户最近正关注这个区别，尚未明确选择新的压力感知触发方案。

### 7.3 历史：独立后段如何继续追围

`usv_env/withdrawal.py::WithdrawalReplay`复制基础env、动力学、波浪、导航记忆、传感器随机流、估计器、消息队列；不复制Renderer与可重建的I1缓存。原训练环境仍保持成功终局，原env.step再次调用仍拒绝。

规则演示继承原RuleController记忆，后段继续五艇动作、真实动力学、通信、传感与配置I1。红船撤退吸引目标为入口以西 `(max(3, exit_x-5), route_origin_y)`，当前(6,30)。WITHDRAW期间不加推进用的USV切向项，但保留原USV径向排斥。撤退参考偏航上限0.4×0.675=0.27 rad/s。

整个红船胶囊越过入口线x=11 m，且x方向速度<−0.02 m/s，才标记withdrawn/verified。中心先过线不足。触礁/碰撞/越界记withdrawal_contact，进入保护区记withdrawal_breakthrough，120 s未完成记withdrawal_timeout。不能因为前段围捕成功就忽略后段失败。

后段复用原围捕门槛作为逐步诊断，记录当前是否有效、最长失效时长、各艇航程、接收track年龄。它不调用原env.step、不计算奖励、不写Replay，也不要求原success始终保持才继续演示。

如果从已训练Actor回放后开启此选项，**后段也切换为规则控制**；不是已训练Actor学出的驱逐行为。必须在展示/论文里说明。

目前没有单独的“驱逐专用控制器”：蓝方延续原规则同侧追堵及AUV跟踪分工。已验证持续跟进，但不能夸大成水面艇全程贴近施压、红船全程被严格包围。

## 8. 已完成的工程工作

### 8.1 本项目此前已完成的主改造

- 从历史六艇结构统一到2USV+3AUV，维度、角色共享Actor、集中式Critic、Replay和日志适配五艇。
- 删除学习槽位链路和相关到点奖励；当前学习输入/奖励依据合法估计和实际实体关系。
- 保留原3-DOF动力学和固定潜深交叉语义，统一碰撞、浅水轮廓、扫掠判定。
- 区分Base接收端hold、传统CV、I2年龄不确定性处理，保持传感器/发包/延迟/丢包共用。
- I1接入执行端、target、Actor前向及I3安全检验；补备用制动可行性并优化计算。
- I3改用完整历史联合动作、语义状态匹配、已实现n-step结果质量、twin保守值与置信增强。
- 加强真实训练、场景验收、通信/协作消融、轻量保存、历史兼容拒绝和逐步诊断。
- 将任务成功改为用户最终指定的三AUV围捕，USV提供辅助阻滞，不把支持门混成成功硬门。
- 按多轮用户明确要求调整三岛间距、红船弧形起步、Y2/Y3离礁余量和C外侧同侧USV出生；本轮没有继续移动它们。

### 8.2 最近一轮具体文件变化

| 文件 | 已有改动 |
| --- | --- |
| `usv_env/red_policy.py` | RedNavigator增加局部切向绕行、选侧锁存、ADVANCE/BYPASS/WITHDRAW、显式撤退请求、历史默认兼容；保留原规则控制器 |
| `configs/scene_template.yaml` | 增加bypass_gain=0.7、withdrawal_yaw_rate_factor=0.4 |
| `usv_env/withdrawal.py` | 新的独立成功后物理演示；蓝方持续规则追围、通信传感/I1、逐步围捕诊断、整艇退出验证 |
| `scripts/visualize.py` | 增加show-withdrawal，原规则回放后传递控制器记忆，后段结束再hold-end，支持headless |
| `usv_env/environment.py` | 记录red_bypass/red_advance事件及渲染行为状态，没改训练终局和奖励 |
| `usv_env/rendering.py` | 兼容新增red_behavior、撤离状态与诊断文字，保留地图风格 |
| `scripts/validate_scene.py` | 轨迹记录增加red_behavior |
| `tests/test_red_withdrawal.py` | 绕行、无计时触发、撤退锁存、终局隔离、整艇越线、继续追围/传感刷新、负终态等测试 |
| `tests/test_visualize.py` | 回放选项检查包含show-withdrawal |
| `validation/five_agent_acceptance/withdrawal/run_replay.py` | 生产env+规则入口跑完整前后两段，写真实CSV/报告/原风格PNG |
| `README.md`、`docs/task_contract.md` | 当前机制、事实边界、原入口命令更新 |
| `validation/five_agent_acceptance/withdrawal/REPORT.md` | 本轮样本、失败/局限、源hash、复现入口 |
| `validation/five_agent_acceptance/forward_patrol/REPORT.md` | 顶部标明其为之前纯APF历史验证，指向当前报告 |

`withdrawal/scope_audit.json`对比前轮同种子配置：只新增两个red_policy配置项并改变task hash；奖励/观测/网络hash、艇位、动力学、训练配置、I1/I2/I3及保存规则一致。没有Git不能伪称掌握所有历史文件的完整diff，以上以当前源码与保存报告核对。

## 9. 当前回放与测试：只引用这些数字

### 9.1 最新场景样本

最新有效目录：`validation/five_agent_acceptance/withdrawal/continued_pursuit/`。全部为Full配置、合法本地信息规则控制，不是学习策略。

| 模式 | seed | 原终局 | 原环境步 / 时间 | 后续撤离步 / 时间 | 接触 |
| --- | ---: | --- | --- | --- | --- |
| 全队 | 5 | 围捕成功 | 321 / 48.15 s | 328 / 49.20 s | 无 |
| 全队 | 3 | 围捕成功 | 319 / 47.85 s | 329 / 49.35 s | 无 |
| 仅AUV，移除两USV | 5 | 红船突破 | 732 / 109.80 s | 无 | 无 |
| 无蓝方 | 5 | 红船突破 | 732 / 109.80 s | 无 | 无 |

全队样本首次BYPASS为42.30/41.85 s。有效正移除贡献审计USV1/USV2分别4/4、4/3，两样本均3次同时正贡献。前段蓝方实体最小净距约0.942/0.758 m。

这支持当前规则样本内水面艇有辅助作用；不证明每艘USV不可替代，不构成统计成功率，也不是新地图下的正式训练结果。以前“USV1贡献一直为零”是旧场景旧规则的问题，当前这两个样本已不再是零，但仍缺多样本/逐艇移除的强结论。

### 9.2 后段追围尚不完整

| 撤离段指标 | seed5 | seed3 |
| --- | ---: | ---: |
| USV1/2航程 | 7.99 / 10.15 m | 9.59 / 10.73 m |
| AUV1/2/3航程 | 16.73 / 18.32 / 16.75 m | 17.02 / 18.87 / 17.26 m |
| 同时满足全部原围捕门槛的步数比例 | 40.85% | 27.96% |
| 最长连续不满足全部门槛 | 27.60 s | 35.10 s |
| 红船整艇撤离验证 | 通过 | 通过 |

seed5离开时最大方位空隙161.84°，略超160°；seed3有一艇5.33 m并有本地观测支持门未满足。两艇/三艇确实继续行动不等于全程严格围住。图上成功绿色文字保留的是前段首次成功事实，不能误读为当前每帧仍成功。

另一个物理/叙事限制：USV追围后程会逐渐落在红船后方，不能仅凭航程非零宣称全程有直接排斥压力。撤退是显式信号导致，不是证明USV把它“逼得只能逃”。

### 9.3 测试及证据身份

最近一次代码变更后全套命令：

```powershell
python -B -m pytest -q -p no:cacheprovider
```

记录为 **255 passed in 38.84s**，日志 `validation/five_agent_acceptance/withdrawal/tests.log`。这是此前实际执行结果；交接本身没有再跑一遍全套，因为仅写文档、未改生产代码。交接时核对当前生产hash仍一致。

配置检查通过；原visualize的HistoryEnv入口headless实际完成前后两段，输出 `capture_success=True withdrawal=withdrawn verified=True elapsed=49.20s`。四份当前回放报告源码hash一致；轨迹行数、终态和六张真实图已核对。

后续若改生产代码，必须重新运行相关测试/回放，不要拿255或旧hash充当新实现验收。历史204/214/246等测试数属于各自当时快照，不能替代当前结果。

## 10. 算法架构与信息权限

| 内容 | 当前实现 |
| --- | --- |
| 算法基线 | MATD3，分散执行、集中训练 |
| 决策数 | 5，索引0/1为USV，2/3/4为AUV |
| Actor | 两个参数集：2USV共享一个，3AUV共享一个；每艇独立用自身合法历史 |
| Critic | 每角色双Critic，分别2头/3头；每个Critic均看全部10维联合动作 |
| Actor历史 | 8帧，每帧90维，展平720维；传感部分75维 |
| Actor网络 | GRU48 + MLP256/256 |
| Critic | 真值/合法历史中央输入，状态98维，历史GRU32 + MLP256/256 |
| Replay | schema16，n-step7，有完整五艇executed动作、端点状态/上下文/phase/activity |
| 观测schema名 | five_agent_local_estimates_no_slots_v2；内部schema版本字符串沿用现代码 |

Actor部署输入：本艇状态、公开地图/保护区、合法本地探测、实际接收到的目标/伙伴消息、时间与mask。未知目标用零值和无效mask，reset不填目标真值或伙伴真值缓存。

集中Critic、训练期I3可使用中央状态；奖励与终止可用真值。rules_local用合法目标估计；rules_ideal必须标真值诊断。红船是独立脚本，不是第六个学习Actor。

AUV传感最大6 m，距离噪声标准差0.2 m，方位3°；探测概率 `1−0.9(d/6)^2`，范围外0。不模拟虚警/声学遮挡，启用未支持项会拒绝。默认延迟3步=0.45 s、独立丢包10%。发送端局部Kalman滤波对各方法共用，不把它当I2独享。

训练seed、每回合scene_seed、独立eval_seed分开记录。当前env按 `config.named_seed` 的scenario、waves、peer_communication、target_communication、sensor+索引命名空间派生随机流；训练用training_scene派生回合场景。复现时用实际保存scene_seed和对应源码，不要把所有随机流当成同一个seed。

## 11. 三个创新模块的当前实现和证据

### 11.1 I1：行动前预测，检查能否安全刹停

通俗说：策略先提出力/力矩，安全层看它未来会不会撞，以及是否还来得及制动；危险则找更安全的执行动作。

当前动作预测20步=3 s，每环境步仍5个子步；之后接原物理受限的纵向/偏航反馈制动，二维速度≤0.02 m/s且偏航≤0.02 rad/s时提前结束，最多200步=30 s。不是每候选固定计算33 s。未能在上限内停住的候选不可标安全，即使净距slack为0。

USV直航0.4 m/s参考制动约4.05 s/0.78 m，AUV0.5 m/s约1.95 s/0.44 m。大横荡/偏航状态USV很难停，离散限值网格最长29.25 s，这也是没有只给1.2 s的原因；网格检查不是整个状态空间证明。

执行路径为nominal→safe→限幅/active mask→executed，Replay保存executed完整动作。target也真实筛选；Actor前向真实筛选、反向近似identity straight-through。I3历史动作只reject不修复。名义不可行时有限网格候选加制动候选，全不可行用已检查的较好者并明确报告fallback/infeasible。

只用合法接收信息；未知艇无法认证，其他艇按常速度预测，波浪不可见扰动未预测。**没有CBF不变集/递归可行性定理，全不可行时没有保证安全的fallback。** 论文应称带备用制动可行性检查的有限候选预测安全过滤。

### 11.2 I1耗时问题的历史和已完成优化

早期300步Base37.02 s、Full472.73 s，约12.77倍；不能一直拿这个当当前速度。

后续同机CPU单线程pilot batch32，300交互步、236次Critic迭代、每角色118次Actor更新，得到：

| 运行 | 墙钟记录 | 边界 |
| --- | ---: | --- |
| 原1.2 s预测未优化重测Full | 199.376 s | 后续同轮对照，不与473直接混算 |
| 仅优化计算，保留旧1.2 s参数Full | 21.711 s | 最终Actor参数SHA与该轮未优化一致，约9.18倍实现加速 |
| 新3 s+备用制动Full | 23.597 s | Base13.772 s，约1.71倍 |
| 顺序复测Full/Base | 23.542 / 13.423 s | 约1.75倍 |

新Full一次记录：执行端I1约0.525 s、target投影4.604 s、Actor投影3.658 s。优化包括Numba编译真实方程/扫掠内核、不启用fast-math、缓存schema索引、Actor公共投影复用、仅重算变化接收艇，以及精确LRU缓存。没关创新、没少更新、没删子步、没靠近似量化放松几何。

这些来自 `validation/five_agent_acceptance/i1_optimization/` 的历史布局短训，不是当前最新场景正式batch256/百万步速度。首次编译、CPU负载、后期I3候选增加都可能改变开销。不能承诺Full固定只有Base的1.75倍。

旧I1失败轨迹：同一冻结Actor，原1.2 s窗口在416步开始不可行，439步USV1触礁；新预测+制动同例588步红船突破、无触礁、最小步末静态净距约0.1655 m。只说明修复了该例安全表现，仍是任务失败，不是安全保证或训练成功。

### 11.3 I2：把旧消息推到现在，并考虑不确定性

通俗说：潜艇发来“刚才红船在这里”，水面艇根据延迟估算它现在在哪里，消息越旧越谨慎，多源信息保守融合。

- Base接收端hold：选择最新原观测时刻track，不外推，不多源融合；仍共用发送端滤波。
- Traditional_CV：最新源做常速度外推。
- I2：按now−t_obs外推位置/速度和协方差，加入过程噪声，按年龄/置信缩放，再用covariance intersection保守融合。
- 去重、同源乱序丢弃，AoI与传输延迟分开；不增加频率、不改消息损伤、不偷真值。

机制和分支测试存在；当前地图下尚无多种子训练证据证明成功率优势。不要只因为I2-on就把所有信息都称实时精确。

### 11.4 I3：训练时参考过去的整队动作

通俗说：遇到类似局面，找过去五艇曾实际执行过的一整套动作；若历史结果可靠、现在也安全且价值更好，就保守用于训练目标。

- 只检索当前在线训练resident Replay；不读演示库、测试轨迹、未来整回合成功标签。
- 一候选是一整条真实 `[5,2]` executed联合动作，五个价值头共享同一个候选，不能从不同历史时刻拼每艇最优动作。
- 默认K8、search_pool512、max_distance2、minimum_quality0.2、增强权重0.5、优势上限1。
- 必须phase、信息valid mask、active mask匹配，元数据有效；语义距离按预先声明物理尺度归一化，不是原始坐标无权距离。
- 质量结合瞬时代理和已完成n-step实际折扣回报的sigmoid置信；已观察窗口包含碰撞/触礁/越界/异常/数值错误时排除。
- 用query端合法信息的预测器reject当前不安全动作；twin目标Critic取min，结合距离、质量、分歧形成置信，给正优势有界增强。
- 无可靠候选、无正优势或terminal bootstrap0时，严格回退原MATD3，增强为0。
- 部署Actor不检索Replay，轻量模型不带历史库。

**历史真实Full短训7552次查询、0合格候选、0实际Bellman增强。** 后续I1优化的短训也仍为0。受控测试有正增强/安全拒绝等证据，但不等于真实训练有效。当前新版场景尚未重做充分训练，因此不能说这个问题已解决，也不能假设只要训久一定解决。

### 11.5 13个实际预设

| 预设 | I1 | I2 | I3 |
| --- | --- | --- | --- |
| Base | 关 | 关 | 关 |
| Full | 开 | 开 | 开 |
| w_o_I1 | 关 | 开 | 开 |
| w_o_I2 | 开 | 关 | 开 |
| w_o_I3 | 开 | 开 | 关 |
| I1_only | 开 | 关 | 关 |
| I2_only | 关 | 开 | 关 |
| I3_only | 关 | 关 | 开 |

另有Traditional_CV与Random_History、No_State_Matching、No_Safety_Screening、Hard_Max四个I3变体。后四者基于Full，对应改变候选/匹配/安全/增强机制；`No_Safety_Screening`指I3历史候选的query安全筛选，不是把整个环境I1关掉。各项应共享地图、奖励、网络容量、信息权限及训练预算。

## 12. 奖励和训练：γ=0.99 预验收后的状态

### 12.1 当前奖励构成

生产入口是 `usv_env/reward.py`，配置在 `configs/scene_template.yaml`，不是用户旧附件的FormationEnvironment奖励。

当前生产入口是 `usv_env/reward.py` 的 `cooperative_pbrs_v1`，公共训练配置为 `gamma=0.99`、n-step=7。AUV用真实距离带、最大方位空隙、凸包边距、合法本地跟踪和20步hold进度构成有界团队势能；本轮 γ=0.99 校准后的 tracking/hold 权重为0.18/0.22。USV只从带符号反事实势能变化和新有效Shapley审计领取信用。旧搜索网格、持续跟踪、独立information/geometry占用和指定点奖励不再支付；旧字段只保留零值兼容列。真正terminal的next potential归零，collector cut保留真实端点。

当前奖励终止量级为：success +3、breakthrough −3、timeout −1.5、碰撞/触礁/越界 −5，scenario_anomaly/numerical_error −10（每active艇）；2026-11-06 按用户要求终局统一除以10，随后过程项外层系数统一乘2.5；gamma=0.99及势能内部比例不变。此前预验收回报和训练均为缩放前尺度。完整公式和参数见 `docs/task_contract.md` 及 `validation/reward_redesign_20261106/REPORT.md`。

### 12.2 新场景奖励审计结论

早期旧终局/旧场景审计发现：持续tracking可支配收益，失败AUV也有很高正回报；一对旧规则轨迹在gamma0.99下，各AUV反而更偏好突破轨迹。已足够说明当时奖励不能冻结，但不能直接声称当前新增围捕进展项后的每条轨迹仍有同样排序。

已在当前地图、动力学和 γ=0.99 下完成13条真实规则/对照轨迹审计，PBRS折扣望远镜最大误差约2e-15；成功规则轨迹折扣团队回报约0.628，持续外圈跟踪、单侧靠近、静止、仅AUV和USV单独拖延均为负。单个seed的排序是机制证据，不是成功率或泛化结论。报告和原始轨迹在 `validation/reward_redesign_20261106/`。

### 12.3 训练状态

- 之前有smoke、300/600步pilot和少量独立评估，用来查更新链路、耗时和失败，不是收敛训练。
- 历史Base/Full seed101各300步，Critic236次、每角色Actor118次；独立评估各2回合均失败，属于旧任务快照。
- 当前 γ=0.99 奖励下 Base/Full 同预算1200步 pilot 均完成真实更新但各只有1个终止回合，分别以reef_collision / breakthrough失败；Base seed4501的2400步短训完成4回合，0次成功（两次触礁、两次突破）。不能拿规则成功或这些短训当作“当前学成”。
- `outputs/Base/seed0`是历史目录；Base/Full还有seed100/seed101。不要覆盖或擅删。
- 默认正式配置batch256、warmup10000、Replay20000、n-step7、gamma0.99；当前短训的轻量Actor约997 KB，未保存Replay/优化器/Critic。
- pilot batch32、warmup64、Replay2000；保留真实场景/物理/奖励，适合链路与早期诊断。
- smoke会缩小网络/Replay等且默认32步外部截断，不能用于研究公平性能比较。

## 13. 项目结构和定位问题的文件

```text
C:\Users\36043\Desktop\shi\
  main.py                  原训练转发入口
  check_config.py          装配配置、维度、Replay内存估计，不宣称策略成功
  requirements.txt
  README.md                当前命令、架构、保存规则和报告入口
  configs\
    default.yaml           正式配置入口，引用scene_template.yaml
    scene_template.yaml    当前场景、物理、奖励、I1/I2/I3和训练参数
    pilot.yaml             短预实验配置
    gamma_099*.yaml         公共gamma对照
  usv_env\
    config.py              严格装配/验证、13预设、hash和命名种子
    environment.py         env.reset/step、感知通信、终止、同状态审计、render_frame
    dynamics.py            原真实3-DOF模型和动作执行
    types.py               ShipState、EnvInfo
    geometry.py/safety.py  实体胶囊、净距、扫掠与保护区路径检查
    scene_config.py        场景合法性
    seeding.py             保留的EpisodeSeeds工具；当前env主要用config.named_seed
    waves.py               原线性波浪
    task.py                真实encirclement_geometry、成功门及辅助门
    reward.py              当前无槽位奖励
    observation.py         Actor合法信息与Critic状态构造
    history.py             HistoryEnv及schema
    communication.py      伙伴消息与目标消息接口
    target_tracking.py    传感、局部跟踪、hold/CV/I2补偿融合
    red_policy.py          RedNavigator和RuleController；文件顶部路线工具供规则/诊断
    safety_projection.py  I1 Python接入、候选筛选、精确缓存
    safety_kernel.py      I1 Numba动力学/几何内核
    rendering.py          原pygame绘图风格及状态显示
    withdrawal.py         独立成功后撤离、蓝方继续追围与验证
  matd3\
    agent.py               MATD3更新、I1训练路径、I3接入
    networks.py            两角色Actor、双Critic与历史编码
    replay_buffer.py       有界resident n-step Replay及联合上下文
    history_enhancement.py I3检索与保守值增强
    checkpoint.py          仅Actor原子保存/加载、兼容验证
  scripts\
    train.py               训练入口、诊断与make_env(HistoryEnv)
    evaluate.py            冻结Actor评估CLI
    research_evaluation.py 评估/通信/协作套件
    visualize.py           Actor、零动作preview、本地规则、可选撤离回放
    validate_scene.py      无蓝方/规则/理想/随机/静止等场景验证
    validate_i1.py         原安全失败例的对照验证
    benchmark_i1.py        安全投影计算基准
    csv_logging.py         原CSV字段
    experiment_data.py    逐步压缩轨迹与诊断数据
    quxian.py              真实已结束回合曲线
    export_policy.py       验证并复制轻量Actor，不分配Replay
    export_comparison.py   汇总各训练种子的评估证据，不把测试回合充作训练种子
  tests\                   当前255项回归覆盖所在目录
  outputs\                 原训练产出，保留历史
  validation\              分轮真实验证与探索记录，注意新旧口径
  docs\
    task_contract.md       公式/信息权限/公平性/保存等详细约定
    HANDOFF.md             本交接
    handoff_state.json     交接状态快照，不是运行配置入口
```

根目录的 `1.html`、`.idea`、缓存等尚在；本轮没有证明它们全部无用，不要凭名字批量删除。

各包还保留 `__init__.py` 作为导入/包标记，不是新的算法模块。

## 14. 反事实分析是干什么的

它不是策略动作时域，也不是撤退触发器。用来估计水面艇究竟有没有阻滞作用，避免只凭“靠近红船”就判贡献。

每10环境步，从同一当前红船状态、导航记忆、波浪与绝对时间出发，持当前蓝方执行力/力矩开环预测20步=3 s，比较全USV、无USV、分别移除USV1/2。沿保护区方向的推进差给总效应、移除效应及两艇Shapley近似。

审计有效且新鲜、正差≥0.01 m才认有效阻滞；两艇移除效应都正才支持共同作用。分支接触、导航异常等会使审计无效。远场可解析为零时不伪造推进量。它只是短时开环近似，不是严格完整闭环因果证明。

长期shadow是无USV的诊断红船，可能混入早先路径差异；不是给蓝方可见的先知，不是第二个敌船，也不算训练样本。反事实和shadow支持部分奖励/诊断，但目前都不直接触发撤离、也不是AUV成功硬门。

## 15. 保存、产出、旧模型与清理

原训练结构：

```text
outputs/<experiment>/seedN/
  latest.pt
  best.pt                  仅真实评估后产生
  train.csv
  updates.csv
  resolved_config.yaml
  run_manifest.json
  diagnostics/             逐步日志/压缩轨迹等
```

`latest/best`仅Actor、必要配置、动作限幅、cruise与元数据；不是完整训练checkpoint。不存Replay、优化器、Critic或仿真器。默认Replay数组估计597,460,000 B，约0.60 GB十进制，另有pending/对象/网络内存；这是RAM开销，不是要写磁盘的大模型。

`--resume`完整恢复明确拒绝。`--warm-start`只在新目录加载兼容Actor，其余状态重建并记录来源SHA/步数。历史六艇、旧目标或schema不兼容要明确拒绝；可兼容历史策略仍要按其保存配置和真实来源解释，不能静默换成新任务的成功模型。

用户要求删垃圾已经做过部分明确死代码/旧学习槽位清理，历史报告有记录；缓存批量删除曾被拒绝，保留。新对话如做清理，应先查引用、区分生产源和历史证据，给具体文件范围，不为“目录好看”删掉反例和失败样本。

## 16. 证据目录的阅读顺序与历史隔离

| 路径（根目录下） | 地位/用途 |
| --- | --- |
| `README.md` | 当前使用入口 |
| `docs/task_contract.md` | 当前实现机制、公式、权限与语义 |
| `validation/five_agent_acceptance/withdrawal/REPORT.md` | 最新围捕/持续追围/撤离样本及局限 |
| `.../withdrawal/continued_pursuit/Full_seed5/report.json` | 当前主要规则示范，含配置、hash、完整前后段信息 |
| `.../withdrawal/continued_pursuit/Full_seed3/report.json` | 第二个完整队伍样本 |
| `.../withdrawal/continued_pursuit/auv_only_seed5/` | 无USV失败对照 |
| `.../withdrawal/continued_pursuit/no_blue_seed5/` | 无蓝方对照 |
| `.../withdrawal/scope_audit.json` | 与前轮配置差异核对 |
| `.../withdrawal/tests.log` | 最新255项回归记录 |
| `.../withdrawal/run_replay.py` | 当前真实验收脚本 |
| `.../i1_optimization/REPORT.md` | I1加速与备用制动实证，历史场景，不是当前训练结论 |
| `.../i1_optimization/performance_summary.json` | 实际计时、Actor一致性及I3零候选数据 |
| `.../forward_patrol/REPORT.md`、`final/` | C外侧部署的前轮纯APF，seed5曾48.00 s，246测试；不是当前48.15 s结果 |
| `.../withdrawal/Full_seed5/`等直接子目录 | 本轮早期蓝方制动停止稿，已被continued_pursuit取代，不能混进最新证据 |
| `.../encirclement/`、`reef_staging/` | 围捕目标/艇位推进过程的历史记录 |
| `.../encounter_review/`、`forward_patrol/channel_deployment/` | 旧内部通道等探索记录 |
| `.../scene_behavior/`、`pure_apf/`、`protected_spawn/`等 | 更早规则/APF/部署验证，口径可能不同 |
| `validation/five_agent_acceptance/REPORT.md` | 2026-11-04五艇初次工程验收、奖励/I3/旧规则失败；部分结论后续已有改进 |
| `validation/five_agent_acceptance/REWARD_AUDIT.md` | 历史奖励反例与量级；新目标奖励要重审 |
| `validation/REPORT.md` | 更早六艇结果，不代表当前任务 |

当前seed5图片在 `validation/five_agent_acceptance/withdrawal/continued_pursuit/Full_seed5/figures/`：start.png、bypass.png、capture_endpoint.png、turning_out.png、outbound.png、withdrawal_endpoint.png。都是原Renderer真实轨迹，不能用生成图替代物理证据。

历史报告记录过备份 `C:\Users\36043\AppData\Local\Temp\shi_before_five_agent_20261104.zip` 和 `validation/development_record/20261104/` 清单；需要回退时先查文件仍在及快照范围，交接未重验该zip，不承诺它涵盖当前改动。

## 17. 常用命令

以下都从原项目根目录运行。普通回放不训练，不需要模型文件。

### 17.1 给用户/老师看的规则演示

```powershell
cd "C:\Users\36043\Desktop\shi"
python -B -m scripts.visualize --rules --experiment Full --seed 5 --speed 1 --show-withdrawal --hold-end
```

当前默认启动即运动，seed5约48.15仿真秒首次围捕，随后同一回合继续撤退与追围，累计97.5仿真秒结束。计算慢时墙钟更长。末帧停留，Esc关闭；Space仅在开局等待或末帧停留时继续，不是运行中暂停键。可加 `--start-paused` 或 `--speed 2`。当前配置有没有 `--show-withdrawal` 都会运行650步同回合逻辑；该选项只为旧立即终止配置追加原独立规则演示。

```powershell
python -B -m scripts.visualize --rules --experiment Full --seed 5 --headless --show-withdrawal
python -B check_config.py --experiment Full --device cpu
python -B -m pytest -q -p no:cacheprovider
```

`--preview`是蓝方零执行器命令的场景预览，不代表规则策略；默认不加preview/rules而无显式checkpoint可能找历史Base/seed0，因此不要让用户误开旧模型。

### 17.2 保存一份新的规则前后段证据

输出目录必须不存在；下面后缀只是示例，实际先检查并选未使用目录：

```powershell
python -B validation/five_agent_acceptance/withdrawal/run_replay.py --seed 5 --experiment Full --mode rules_local --output validation/five_agent_acceptance/withdrawal/recheck_seed5 --render
```

同脚本支持 `--mode auv_only`、`--mode no_blue`，以及Base/Full。报告中的capture_time_s对失败模式其实是原回合结束时间，读时要同时看outcome，不能把失败耗时称完成任务时间。

### 17.3 短训练和正式训练只是后续入口

交接没有启动以下训练。当前重点是先明确场景语义、复核奖励与I3，再按用户后续要求执行。新run必须用未使用目录和种子，不能覆盖历史。

```powershell
python -B -m scripts.train --config configs/pilot.yaml --experiment Base --seed 3100 --steps 600 --runs-root outputs/next_pilot --device cpu --eval-every 0
python -B -m scripts.train --config configs/pilot.yaml --experiment Full --seed 3100 --steps 600 --runs-root outputs/next_pilot --device cpu --eval-every 0
```

600步仅查链路与早期诊断，不称收敛。完整训练的既有命令在README，勿把“有命令”解释为已运行。GPU训练可用 `--device auto/cuda`，但I1有CPU预测、首次Numba编译等成本，不能直接按GPU算力外推加速。

冻结评估必须显式指定实际存在的当前兼容Actor和独立评估种子，例如：

```powershell
python -B -m scripts.evaluate --checkpoint outputs/next_pilot/Base/seed3100/latest.pt --episodes 30 --base-seed 2000000 --variant fixed --suite nominal --output validation/next_base_eval.csv
```

这是路径示例，不表示该文件已产生。通信/协作套件及曲线导出见README。`quxian`只用真实已结束回合；没有完整回合不能插值造学习曲线。

## 18. 接下来建议怎么做：事实与建议分开

### 18.1 最接近用户当前关切：撤退触发叙事

当前围捕确认→撤退是已经落实并验证的口径。用户刚发现它不直接依赖USV阻截，可能希望进一步推敲，但还没有明确提出新触发条件。

建议新对话先解释清楚：训练成功仍为AUV围捕，后续继续追围已经实现；若要“红船自己认输”，需要设计可观测的任务放弃机制，而不能简单看到AUV真值/计时到点就逃。不要未讨论就把USV贡献加进成功硬门，也不要私自把终局改成守区或撤离。

如果用户要求探索压力驱动撤退，可考虑红船自身的受阻推进、局部可行通道、近距离水面威胁和持续时间，辅以锁存/迟滞避免反复掉头；这是候选思路，尚未采纳、没有实现和证据。必须同时检查无蓝方、只有USV、只有AUV、全队，避免红船无缘无故撤退或规则放弃过早使任务变简单。保留用户已确认的训练成功定义，除非其明确修改。

### 18.2 场景只做必要收尾

- 出生位置已基本认可，不从头洗牌。
- 若用户要继续改追围，先看现在失围的具体原因：观测年龄/误差、转向与加速度、AUV速度余量、USV是否落后等，区别几何与信息门失效。
- 用户已叫停规则精修，不为了“100%成功”无限扫参数/种子，也不通过放宽160°、5 m或3 s来消灭失败。
- 可以补少量独立种子与明确逐艇对照来检查水面参与性，但不能把两条好看轨迹当全场景性能。

### 18.3 真正进入强化学习前优先审奖励

已完成当前规则、静止、持续跟踪、单侧靠近、无USV、USV-only、AUV-only和事故轨迹审计；后续所有方法共享 γ=0.99 和当前 reward hash。若奖励或场景再次修改，必须新建验证目录并重新记录hash，不能复用旧数值。

### 18.4 重点让I3有可解释的实际工作证据

用足够诊断的真实pilot分层统计：phase匹配、info mask、active mask、元数据、质量阈值、距离、安全拒绝、优势非正、最终增强。先找哪一层挡掉候选，再决定是否改设计。不要为让日志非零直接降低质量门槛、混入规则示范或偷用最终测试标签。

若依然长期零增强，坦诚说明机制不适配当前训练分布，先研究或与用户讨论创新取舍；不能仅用受控测试证明“已经有效”。

### 18.5 公平pilot、正式训练与安全复核

奖励与语义基本稳定后，Base/Full及必要消融用同种子、网络容量和交互/更新预算先跑真实pilot；观察是否有完整回合、围捕进展和安全改善，再决定多种子长训。固定训练/调参/最终测试种子用途，别用终测挑奖励/模型。

I1已明显提速，应先测当前数据分布的成本再决定替换。保留机制并记录环境/target/Actor/I3时间，区分首次编译和稳定运行；用更广失败例审查不可行与备用制动局限，不把通过一条旧触礁重放说成严格安全。

## 19. 接手时最容易犯的错误

| 容易误说/误做 | 正确口径 |
| --- | --- |
| 规则成功，所以RL已经成功 | 规则只证明这些样本可达，当前没有学成策略证据 |
| 现在还是纯APF | 当前APF+切向绕行；历史bypass_gain缺失/0才保留纯APF |
| 蓝船一挡红船就退 | 当前由AUV围捕成功+开启后段触发 |
| 红船没USV也会按时间撤退 | 没有计时撤退；无围捕不触发 |
| 撤退后蓝方停下 | 当前已改为继续规则追围，早期停止稿只属历史 |
| 五艇在动，所以全程有效包围 | 当前后段有效门占比仅40.85%/27.96%，有长时间失效 |
| 红船越过地图外边界才完成撤离 | 当前是整艇越过入口线x=11，仍在55×55地图内 |
| I1是有保证的CBF | 有备用制动检查的有限候选预测筛选，没有严格保证 |
| Full永远慢12.77倍 | 旧数据；后续旧布局300步基准约1.7–1.75倍，仍不能保证新长训比例 |
| I3接上代码就有效 | 真实短训合格候选0，增强0，当前仍待验证 |
| 五个积分子步等于五次RL动作/更新 | 它们属于同一环境步的内部积分 |
| 只守住保护区就是成功 | 必须三AUV围捕连续3 s |
| 必须两USV都有贡献才算成功 | 当前只是辅助诊断，不是成功硬门 |
| user旧奖励附件就是当前函数 | 它属于旧编队任务，当前入口usv_env/reward.py |
| 直接跑默认visualize就是现在规则 | 要显式--rules；默认可能找历史checkpoint |
| 旧报告所有No-Go都还原样成立 | 分清后续I1/艇位/围捕改进与仍未解决的奖励/I3问题 |
| 新源码改完还引用旧回放hash | 必须重做相关验证并保留旧证据身份 |
| 清垃圾等于删所有validation | 历史报告和失败反例是科研证据，不能批量丢掉 |

## 20. 可直接粘贴到新对话的开场说明

```text
请接手 C:\Users\36043\Desktop\shi 的成熟科研工程。先完整阅读 docs\HANDOFF.md，结合 docs\handoff_state.json、docs\task_contract.md 和 validation\five_agent_acceptance\withdrawal\REPORT.md 核对当前状态。

必须在原工程精准修改，保留架构、入口、输出目录、原地图风格、欠驱动3-DOF与轻量latest/best；不保存Replay/优化器/Critic/仿真器大checkpoint，不覆盖旧结果。禁止Superpowers，不默认启动子代理。学习观测/奖励严禁槽位和参考点追点，规则导航允许。

当前为2USV+3AUV；USV同侧阻截给AUV争取时间，三AUV有效围捕连续3秒即确认首次capture。当前默认首次capture后立即触发红船向入口撤退，五蓝艇继续当前学习/Actor/规则策略、传感通信和配置I1，后段进入奖励和Replay；到本局累计650步（97.5秒）正常结束，不是成功后另加650步。未capture仍900步timeout，晚capture当步结束，真实安全事故仍提前终止。旧独立规则演示仅兼容缺少新字段或显式为0的历史配置。

现代码由AUV首次围捕成功给capture_confirmed信号；当前默认不依赖--show-withdrawal，也不是红船自主判断USV压力。没USV但AUV仍成功也会退；没围捕不退。不要另行实施压力触发机制。查看validation/post_capture_650_20261106/当前证据，历史独立后段的成功率和hash不代表本轮学习结果。

最新规则seed5/3分别48.15/47.85秒围捕，再49.20/49.35秒完成整艇越过入口，无接触；撤离时严格围捕门只在40.85%/27.96%的步数成立，不能说全程围住。无USV及无蓝方seed5均109.8秒突破。当前255项测试通过记录与源码hash在交接里；成功画面是规则，不是RL已学会。

算法仍I1预测安全+备用制动、I2旧消息年龄/不确定性补偿、I3历史完整联合动作保守增强。I1无严格CBF保证，后续300步旧布局耗时已降至Full/Base约1.7倍，不能沿用早期12.77倍。I3历史真实短训7552查询却0合格/0增强，尚未解决。当前地图未完成正式训练，奖励和gamma需新场景审计。

先读现有证据，不重搭工程、不无限精修规则、不擅自启动百万步训练。需要继续做的重点、代码定位、复现命令和历史隔离都在HANDOFF.md。解释尽量通俗，约每60秒汇报进展。
```

## 21. 650步任务训练前修复（覆盖上面的旧状态）

用户已明确授权“发现问题直接修改”，目标为尽快开展有效训练。最新源码、验证、实际短测及限制以 `validation/training_ready650_20261106/REPORT.md` 为准；上面的只读结论、旧维度、旧默认模型路径和旧时间估计不再代表当前版本。

- 围捕锁存后，将已经失效的USV阻截gap和Shapley支付项置0，其余奖励权重与非正过程成本不变；reward.phase_contract=capture_support_v1进入hash。
- 新观测schema为five_agent_local_estimates_capture_v3：sensor76、frame91、Actor728、Critic99。新增公共capture_confirmed，时间按未捕获900/已捕获650的实际截止计算。通知采用全艇即时可靠任务事件假设，所有实验一致；不提供目标定位或规则动作。旧720维Actor拒载，原暂停权重保留。
- 评价新增首次capture率、首次捕获步/秒、围捕后安全成功率、整艇退出率；区分全部回合与已捕获回合分母。原success_rate和best选择规则不变。
- 新训练使用configs/fast.yaml的I3 256/4，输出到outputs/formal650_v1。visualize默认读取该目录下Full/seed0/latest.pt。10000步热身、batch256、一步一更Critic、Actor延迟2、gamma.99、物理积分和原围捕条件均保留。
- 新增scripts/cloud_preflight.py，云端先执行实际GRU/MLP CUDA前后向检查。NUMBA_NUM_THREADS=3配合OMP/MKL/OPENBLAS=1用于线程预算，PyTorch仍1线程；它不绑定CPU，也不能保证五进程比两进程快。
- 两种混合历史GRU优化均实测更慢，未合入生产。不要将负结果原型替换当前networks.py。I3懒安全筛选尚仅做只读设计评估，未落地，不得把它当作已有加速。
- 本轮只启动有界诊断，没有启动1000回合长训。规则与有限梯度更新不证明RL收敛；云卡耗时与最佳并发数仍需在租机实测。不要沿用旧25–43小时或声称大半天必定完成。

## 22. 简单nohup入口与3000回合默认配置（2026-11-07）

用户要求把启动配置内置，保留五条 `nohup python -u scripts/train.py ...` 命令形式。训练入口现在默认加载configs/fast.yaml（I3 256/4），轨迹每10局采一局、分项诊断每100步。直接执行脚本或模块时，在数值库导入前固定NUMBA_NUM_THREADS=3、OMP/MKL/OPENBLAS=1；导入train模块不改调用方环境。PyTorch仍按原配置1线程。

训练默认输出改为outputs/formal650_3000ep，visualize默认读取其中Full/seed0/latest.pt。配置、输出目录和日志频率仍允许显式参数覆盖；全局assemble()默认仍是原512/8，未修改。每个实验使用 `--episodes 3000 --device auto`，不要同时传旧的 `--steps 10000000`。3000是完成的训练回合数，包括热身期间结束的回合；热身仍10000环境步。所有更新频率、网络、奖励、物理和成功规则均未在这次入口修改中调整。

五条完整Linux命令在README开头。先将更新后的工程同步到云机，再运行；本地修改不会自动同步。目录非空仍拒绝覆盖，不自动续训或删除旧数据。本轮没有启动正式长训，5进程共享GPU也不保证提速。

本次入口相关37项通过；完整回归为448通过、11失败。11项全部由于validation/runtime_quality_20261106/baseline/下的history_enhancement.py和safety.py历史对照文件缺失，未执行到旧新结果比较；没有用当前实现伪造历史对照，也没有跳过/删除这些测试。现工作区此前的validation报告目录和outputs也已不存在，上一节历史报告与旧权重保留描述不可当作当前文件存在证明。详见validation/simple_launch_3000ep_20261107/REPORT.md及tests.xml。
