# 最终功能图图注（中文）

## 图 1｜最终证据边界框架

a-d 分别给出 path/source-ID/MD5 互斥的数据角色、2 s Log-Mel 共享层级卷积循环神经网络（CRNN）与推理角色。B2 Raw Softmax 是主要四类路由；并行的冻结嵌入进入仅训练集主类/子类原型，输出 Top-k 候选、距离、预测 Top-1/Top-2 距离间隔、层级一致性与原型代表；校准仅使用验证集，B3 仍为消融。n 不适用；均值、SD、CI、Wilcoxon 与 Holm 均不适用。协议要素原则上可部署，但本图未证明猪个体、录音会话、设备或农场层面的分组独立性，也不证明准确率或外部有效性。

## 图 2｜验证选择的主要分类证据

a 在 0-1 纵轴上显示全部 n=25 个运行点及 Macro-F1 均值±SD；b 显示恰好三个匹配增量及 fold-cluster bootstrap 95% CI。B1−B0：P=0.000162303448，Holm 不适用（预先设定的时长对比）；B2−B1：P=0.122846338，Holm P=0.614231688；B3−B2：P=0.477536357，Holm P=0.955072713。B2/B3 保留锁定的七比较 Holm family。B1 是主要支持增益；B2 为探索性且不显著；B3 不改善 Top-1。c 显示逐 fold 的验证选择 λ，未使用测试集。这些标签感知检验不构成可部署性能保证。

## 图 3｜原型功能价值

a 报告主类/子类 Rank-1 与 Rank-2 覆盖，b 报告 feeding/stress Top-2 覆盖。c 分别重算 Raw、Main、Hierarchical 的不一致率、错误精确率、错误召回率与富集比；无标志运行的富集比不可定义（Raw n=24, Main n=22, Hierarchical n=18）。d 中 Main 与 Hierarchical 共享同一个四主类原型的第二近减最近距离，只按各自正确/错误划分；不存在层级余弦距离向量。总运行数 n=25，必要时显示指标特定的可用运行数；中心为 fold 均值的平均，离散为可用运行 SD，区间为 fold-cluster bootstrap 95% CI。无新增 Wilcoxon，Holm 不适用。间隔与不一致标志在使用时可部署，正确性、精确率、召回率和 AUROC 为标签感知评估。低流行率与低召回率限制了该标志，不证明安全自动拒识。

## 图 4｜确定性候选案例

fold 0/seed 42 的 7 个角色为：C1 cough 正确例、C2 calm-grunt 正确例、C3 feeding 正确例、C4 stress-vocal 正确例、C5 feeding 边界例、C6 stress 边界例及 C7 低预测间隔例。d 使用层级路由预测的主类；原型代表样本定义为 a training sample closest to the predicted class prototype。n=7；均值、SD、CI、Wilcoxon 与 Holm 不适用。预测距离间隔在使用时可部署，案例正确性为标签感知；案例不估计总体流行率。

## 补充图 S1｜完整干净消融

每行 n=15 或 n=25，以均值±样本 SD 表示；不匹配行无 fold-cluster bootstrap 95% CI、Wilcoxon 或 Holm 推断。标签感知结果不证明替代前端具有可部署优势。

## 补充图 S2｜固定 λ 回顾性累积证据

a 显示全部 n=25 条 fixed-lambda=0.5 fold-seed 轨迹及其均值；b 为配对均值差与存储的 10,000 次 run-bootstrap CI、五个未进行多重性校正的双侧 Wilcoxon 数值 P 值（B1 - B0 P=0.000162303448; B2 - B1 P=0.170241396; B3 - B2 P=0.0582529521; B3 - B1 P=0.0138085615; B3 - B0 P=2.98023224e-07）。该回顾性族未用 Holm；这是描述性、标签感知的运行层面分析，不替代验证选择的主要分析，也不证明可部署增益。

## 补充图 S3｜逐 fold 的 λ 选择

每个候选在每个 fold 有 n=5 个种子验证最优值，显示均值±SD。选择不使用测试值，也不使用 fold-cluster CI、Wilcoxon 或 Holm；不证明全局最优 λ。

## 补充图 S4｜最佳 epoch 与验证-测试差

每阶段 n=25，以中位数与四分位距表示；未新增均值±SD、fold-cluster CI、Wilcoxon 或 Holm。差值为标签感知诊断，不允许测试集调参。

## 补充图 S5｜DEMAND 模拟噪声

受控加性 DEMAND 噪声下 n=25，显示均值±SD。DWASHING 为含前开式洗衣机运转声的家庭洗衣间录音，TBUS 为公共交通巴士录音，STRAFFIC 为繁忙街道交通路口录音；d 为存储的 fixed-lambda=0.5 ALL_NOISY 聚合。未新增 fold-cluster CI、Wilcoxon 或 Holm。标签感知性能不证明真实猪场外部有效性。

## 补充图 S6｜AURC 与 AUGRC

ALL_NOISY 的风险-覆盖曲线下面积（AURC）与广义风险-覆盖曲线下面积（AUGRC）为描述性 n=25 均值，越低越好；无可用 SD、fold-cluster CI、Wilcoxon 或 Holm。排序分数在使用时可部署，风险结局为标签感知；不证明通用工作点。

## 补充图 S7｜标签感知真类间隔

n=25；中心为运行中位数的均值，变异性由图中的 fold-cluster bootstrap 95% CI 表示（未编码 SD）；无新增 Wilcoxon 或 Holm。该量使用真实标签、不可部署，也不是层级决策距离。

## 统计与证据约定

除另有说明外，n=25 表示匹配的 fold-seed 运行；中心为 5 个 fold 均值的平均，离散度为可用运行的样本 SD。区间为先在 5 个 fold 内平均种子后进行的 fold-cluster bootstrap 95% CI。配对检验使用 SciPy method=auto 的双侧 Wilcoxon 符号秩检验；存在零差时使用渐近 P 值。报告数值 P 值与适用的 Holm 校正，不使用显著性星号。可部署评分或标志在使用时不依赖真实标签；正确性、错误率、AUROC、精确率、召回率与真类间隔均属于标签感知评估。证据不证明真实猪场外部有效性或自动路由。
