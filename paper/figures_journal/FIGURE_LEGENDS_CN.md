# 图注（中文）

## 图 1 | 有证据边界的猪声分类框架
本图回答锁定的干净与模拟噪声评估流程如何组织。**a**，公开/第三方猪声音频表示为 2 s 波形，经 Log-Mel 特征和 CRNN 编码。**b**，四分类主头与六子类型辅助头并行监督共享表示。**c**，主类和子类型原型仅由训练嵌入构建，校准仅使用验证集；raw Softmax、main prototype 和 hierarchical prototype 独立评估，不存在自动路由。**d**，冻结的干净评估与 DEMAND 模拟加性噪声评估分开。本图为协议示意图，n 不适用，且不涉及误差线、推断检验或 P 值。本干净/协议图不使用 active-event SNR；其他图中，active-event SNR 按干净事件的有效非填充区域计算。本图不能推出真实猪场外部验证，也不能推出新的自动路由机制。源数据对应关系见 figure source map。

## 图 2 | 时长选择与锁定的干净消融
本图回答为何 2 s Log-Mel 成为干净主线。**a**，1 s（n=25）、2 s（n=25）和 3 s（n=15）的 Macro-F1 均值 ± 样本标准差；n 是 fold-seed run，25=5 folds×5 seeds，15=5 folds×3 seeds。**b**，n=25 个匹配 fold-seed 对的 2 s−1 s 差值；点为平均差 0.024631，区间为 10,000 次配对差值百分位 bootstrap 95% CI [0.014113, 0.035642]，双侧 Wilcoxon 精确结果为 P=0.000162303。**c**，锁定的干净替代方案显示描述性均值 ± 样本标准差并逐行标 n；n=15 与 n=25 不作为配对检验。本干净/协议图不使用 active-event SNR；其他图中，active-event SNR 按干净事件的有效非填充区域计算。本图不能推出 3 s 显著差于 2 s，也不能推出 2 s 普遍最优。

## 图 3 | 层级监督与干净原型推理
本图回答层级结构在不过度声称干净性能时提供什么。**a**，四个主类对应六个辅助子类型。**b**，λ=0.2、0.5、1.0 在 n=25 个 fold-seed run 上的 Macro-F1 均值 ± 样本标准差；λ=0.5 的标准差最低，λ=1.0 的均值最高。相对匹配的非层级 2 s baseline，双侧 Wilcoxon P 值依次为 P=0.265132、P=0.170241、P=0.153127，所有配对 bootstrap CI 均跨 0。**c**，raw Softmax、main prototype 与 hierarchical prototype 来自同一 λ=0.5 checkpoint cohort（n=25），误差线为样本标准差；main 对 raw 为 P=0.571243，hierarchical 对 raw 为 P=0.058253，均为未校正双侧 Wilcoxon。本干净/协议图不使用 active-event SNR；其他图中，active-event SNR 按干净事件的有效非填充区域计算。不能声称层级监督或干净 hierarchical prototype 带来显著增量。

## 图 4 | 主类原型在模拟加性噪声下的表现
本图回答原型鲁棒性在哪些条件下得到支持，以及聚类敏感性如何改变推断。**a**，moderate noise（20/10 dB）、extreme stress（0 dB）和全部 9 个噪声条件的 n=25 fold-seed 分层均值，显示 Macro-F1 均值 ± 样本标准差。**b**，DWASHING、TBUS、STRAFFIC 的 clean/20/10/0 dB 均值轨迹，本 panel 不画区间。**c**，main−raw、hierarchical−raw、hierarchical−main 的配对差值；实心圆/粗线为 25 对 run-level 的 10,000 次 bootstrap CI，空心方块/细线为先在 5 个 fold 内平均 seeds 后的 10,000 次 fold-cluster bootstrap CI。每行均标源文件中保存的 exact run/fold Wilcoxon P 值，未作多重性校正。main−raw 在 moderate、ALL_NOISY、extreme 的 run/fold P 分别为 P=0.00167307/P=0.0625、P=0.0105108/P=0.3125、P=0.560171/P=1。active-event SNR 按干净事件的有效非填充区域计算。0 dB 是极端模拟噪声压力条件，不是无噪声条件。证据在 moderate noise 最强；不得脱离 fold-cluster 敏感性单独宣称 run-level 显著，也不得声称 hierarchical prototype 在噪声下最稳健。

## 图 5 | 失败模式与选择性预测
本图回答剩余失败是什么，以及 prototype 是否改善不确定性排序。**a**，每个 DEMAND 环境从 clean 到 20/10/0 dB 的 pooled-confusion cough recall。**b**，ALL_NOISY 下 feeding→stress 与 stress→feeding 的 pooled decision counts。Panels a 和 b 汇总同一组 25 个 fold-seed runs 的重复决策，这些决策不是独立录音。**c**，ALL_NOISY 的 AURC、AUGRC、risk@95% coverage 和 frozen-threshold selective risk，均为描述性 n=25 均值，其中 n 表示 fold-seed runs；越低越好，没有可用误差线或配对推断 P 值。raw Softmax 的 AURC/AUGRC 更好；main prototype 仅在固定工作点风险上有小优势。active-event SNR 按干净事件的有效非填充区域计算。0 dB 是极端模拟噪声压力条件，不是无噪声条件。本图不能推出 0 dB cough 已可靠、错误具有因果机制，或 prototype 普遍改善不确定性。

## 补充图 S1 | 完整锁定干净消融
本图回答任何锁定的干净架构/前端变体是否替代 2 s Log-Mel 主线。点为 run-level 均值，粗线为均值 ± 样本标准差，细线为观察到的 min–max。n=25 表示 5 folds×5 seeds，n=15 表示 5 folds×3 seeds。各行没有匹配推断检验或 P 值。本干净/协议图不使用 active-event SNR；其他图中，active-event SNR 按干净事件的有效非填充区域计算。不加入仅归档但未锁定的数值，也不把 unmatched rows 解释为显著性证据。

## 补充图 S2 | DEMAND 环境 × SNR 详细结果
本图回答每种推理方法在所有 DEMAND environment×SNR 组合中的表现。**a–d** 分别为 raw Softmax、main prototype、hierarchical prototype 和 fused 消融。点为 n=25 fold-seed 均值，误差线为样本标准差，虚线为对应 clean 均值。没有 multiplicity-adjusted environment-specific inference 或 fold-cluster P 值。active-event SNR 按干净事件的有效非填充区域计算。0 dB 是极端模拟噪声压力条件，不是无噪声条件。这些受控模拟不是真实猪场外部验证，fused 也不是主贡献。

## 补充图 S3 | 混淆、校准与 fold 敏感性
本图回答 pooled 错误、验证集校准和 fold-level 效应是否支持主张。**a**，四种方法的完整 ALL_NOISY 4×4 混淆矩阵，逐真实类别行归一化为百分比，气泡面积表示比例；每个真实类行汇总 25 个 fold-seed run 与 9 个噪声条件的 9,450 个重复决策。**b**，n=25 次验证集校准的完整离散参数选择频率。**c**，同一 n=25 验证校准中的全局与逐类冻结拒绝阈值分布。**d**，三个分层与三种比较的全部 5 个 fold 均值差；点为 fold，横线为五 fold 均值。本图不新增推断 P 值；exact run/fold P 保留在图 4/source data。active-event SNR 按干净事件的有效非填充区域计算。0 dB 是极端模拟噪声压力条件，不是无噪声条件。校准仅使用验证集，5 个 fold 点不能当作 25 个独立 run，pooled decisions 也不是独立录音。
