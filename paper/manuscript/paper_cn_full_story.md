# 面向模拟环境噪声的猪声识别：时长感知层级表征与后验原型推理

## Abstract

规模化养殖中的猪声识别不是单纯的分类器替换问题。猪声事件往往包含起始瞬态、持续声学纹理和短时上下文，过短的输入窗口可能截断 feeding 或 stress_vocal 的上下文，过长的窗口又可能引入冗余背景。本文在已经锁定的 class-capped cap3x、2 秒 Log-Mel、CRNN 主干基础上，将论文主线重构为三个有边界的科学问题：为什么先确定事件上下文，为什么引入层级辅助监督，以及为什么在噪声下使用后验原型推理。任务包含 cough、calm_grunt、feeding 和 stress_vocal 四个主类，并使用 dry_cough、abdominal_cough、calm_grunt、feeding、frightened_stress 和 anxious_stress 六个子类型作为辅助监督。干净闭集实验表明，2 秒 Log-Mel CRNN 的 Macro-F1 为 0.9472，高于 1 秒 Log-Mel 的 0.9226，配对平均增量为 0.0246，Wilcoxon p=0.000162。MCTAFD、PCEN、spectral-gate denoising、SpecAugment 和 attention pooling 等复杂改动没有形成比 2 秒 Log-Mel CRNN 更稳定的干净主线。层级辅助监督在 lambda=1.0 时取得最高均值 Macro-F1 0.9515，lambda=0.5 时标准差最低并作为平衡设置，但其相对 2 秒基线的增益未达到统计显著，因此本文将其定位为细粒度语义和边界解释贡献，而不是强性能贡献。后验原型推理在干净条件下以 hierarchical prototype 取得最高 Macro-F1 0.9540；但在 DEMAND 模拟加性噪声下，main prototype 更稳健，在 ALL_NOISY 条件下 Macro-F1 为 0.6230，高于 raw Softmax 的 0.6173，平均增量 0.005755，95% bootstrap CI [0.001606, 0.010894]，Wilcoxon p=0.010511。本文不声称真实猪场外部验证，不把 0 dB active-event SNR 解释为无噪声，不声称 0 dB 下 cough 已可靠识别，也不声称 prototype 全面改善不确定性。

## Keywords

猪声识别；Log-Mel；CRNN；层级监督；声学原型；模拟加性噪声；DEMAND；选择性预测

## Introduction

猪声自动识别可以为养殖场健康监测和行为感知提供非接触入口。与图像或结构化传感器相比，声音信号更容易受到设备噪声、交通噪声和环境混响影响；但在早期闭集建模阶段，一个同样基础的问题是：模型应该看到多长的猪声事件。cough、calm_grunt、feeding 和 stress_vocal 并不是完全孤立的瞬时标签。feeding 和 stress_vocal 往往需要短时上下文才能区分，cough 也包含起始和衰减结构。因此，本文的第一条主线不是把 2 秒输入包装成新颖性口号，而是用配对证据说明，干净主线应该先从合适事件上下文开始，而不是继续堆叠特征。

这一判断来自失败证据。1 秒 Log-Mel CRNN 的 25-run Macro-F1 为 0.9226；2 秒 Log-Mel CRNN 提升到 0.9472，配对平均增量 0.0246，Wilcoxon p=0.000162。3 秒 Log-Mel 在可用 15-run 比较中为 0.9434，没有超过 2 秒设置。更复杂的改动也没有形成更稳定主线：cap3x mctafd_no_se_gated 为 0.9208，logmel_pcen 为 0.8916，logmel_sg 为 0.9093，logmel_specaug 为 0.9198，logmel_dur2_attn 为 0.9431。BiLSTM replacement 在项目记录中被保留为已完成但非主线方向，当前主表不重复其具体数值。由此，clean 主线从“堆特征”收束为“先找对事件上下文”。

第二条主线来自标签粒度。四个主类适合部署报告，但对声学差异来说仍然粗糙。cough 内部有 dry_cough 和 abdominal_cough；stress_vocal 内部有 frightened_stress 和 anxious_stress；feeding 与 stress_vocal 的边界长期出现在 confusion 方向中。本文因此把四类主任务与六类子类型任务绑定，采用 `main loss + lambda * aux loss` 的训练目标。lambda=1.0 的均值最高，Macro-F1 为 0.9515；lambda=0.5 的标准差最低，为 0.0124，因此作为平衡设置进入原型和噪声实验。需要强调的是，层级辅助监督相对 2 秒基线的增益未达到统计显著，不能写成显著性能提升。它的作用是提供细粒度语义约束，并帮助解释 feeding/stress_vocal 等边界问题。

第三条主线来自噪声推理。Raw Softmax 表示清洁训练条件下学到的判别边界；加性噪声会移动 embedding，使边界附近的样本更不稳定。后验原型推理不重新训练模型，而是在每个 fold-seed 内用 train embedding 构建类别中心，用 validation 校准温度、阈值和融合权重，用 test 做冻结评估。main prototype 只使用四个主类中心；hierarchical prototype 进一步加入六个子类原型。干净条件下，hierarchical prototype 最好，Macro-F1 为 0.9540；但 DEMAND 模拟噪声下，main prototype 最稳。这一反差构成本文的核心解释：层级原型适合干净语义解释，主类原型更适合中等噪声鲁棒推理。

## Related Work

猪声识别属于动物声学监测和监督式音频分类的交叉问题。最终投稿稿件需要补充经过核验的猪声识别、畜牧声学监测和农场环境噪声文献；当前参考文献文件只提供 DEMAND、CRNN、prototypical networks、selective classification、librosa 和 fd-shifts risk-coverage 实现等保守条目。本文不通过贬低既有工作来制造创新性，而是把区别放在三个可审计环节：事件时长选择、层级子类型监督和冻结后验原型推理。

CRNN 已广泛用于音频分类，其卷积层提取局部时频结构，循环层整合时间信息。本文不把 CRNN 作为新模型贡献，而把它作为稳定主干，用于验证输入时长和监督结构是否真正被猪声证据支持。Log-Mel 是可复现的基础声学表示；结果表明，在当前 clean 包中，PCEN、spectral-gate denoising、SpecAugment 和 MCTAFD 类复杂改动均未超过 2 秒 Log-Mel CRNN 主线。

原型方法与 prototypical networks 相关，但本文不声称提出 prototype learning。这里的 prototype 是后验推理层：模型训练完成后，在每个 fold-seed 的 train split 上构建 embedding 类中心，在 validation split 上校准，在 test split 上冻结评估。Selective prediction 和 risk-coverage 用于衡量置信度或距离分数是否能排序错误样本。本文的 corrected AUGRC 按 fd-shifts generalized risk 定义从保存的预测 CSV 重新计算，因此不能沿用旧的 AURC=AUGRC 结果。

## Materials and Methods

### 任务和标签

闭集任务包含四个主类：cough、calm_grunt、feeding 和 stress_vocal。层级辅助任务包含六个子类型：dry_cough、abdominal_cough、calm_grunt、feeding、frightened_stress 和 anxious_stress。子类型只用于辅助监督和原型解释，不改变最终四类输出。

### 干净主线

干净主线由 class-capped cap3x 扩展、2 秒 Log-Mel 输入和 CRNN backbone 组成。1 秒、2 秒和 3 秒分别对应三种假设：1 秒是否足够紧凑，2 秒是否提供必要事件上下文，3 秒是否继续获益或引入冗余背景。结果支持 2 秒作为当前包内最稳健的 clean 主线。

### 层级辅助监督

层级模型在四类主分类头之外加入六类子类型头。训练目标为 `L = L_main + lambda * L_aux`。lambda=0.2、0.5 和 1.0 均完成 25-run 汇总。lambda=1.0 均值最高，lambda=0.5 标准差最低。本文采用 lambda=0.5 作为平衡设置，同时报告 lambda=1.0 具有最高均值，避免把稳定性选择和最高均值选择混为一谈。

### 后验原型推理

原型推理在模型训练完成后执行。每个 fold-seed 独立使用 train embedding 构建原型，不跨 fold 或 seed 混合。main prototype 构建四个主类中心；hierarchical prototype 进一步构建六个子类中心。validation split 只用于温度、阈值和融合权重校准；test split 只用于冻结评估。该模块不修改 clean checkpoint，也不使用 noisy validation 调参。

### DEMAND 模拟噪声协议

噪声实验使用 DEMAND 的 DWASHING、TBUS 和 STRAFFIC 三类环境录音。混合强度为 active-event SNR 20、10 和 0 dB。这里的 0 dB 是活动事件区域 SNR 的极端压力条件，不是无噪声。协议为 zero-shot frozen：复用 clean checkpoint、train-only prototypes、validation-only calibration 和 frozen thresholds。噪声片段选择由 SHA256 key 决定，并固定 global_noise_seed=3407、noise_repeat=0。cross-seed draw audit 共 2520 行，failed_rows=0，说明同一 clean MD5、fold 和环境在不同模型 seed 下复用了相同噪声片段。

## Experimental Protocol

最终 clean 和 simulated-noise 结果均按 5 folds x 5 seeds 汇总。模型比较使用 matched fold-seed paired statistics。训练数据可用于模型参数和原型构建，验证数据可用于温度、阈值和融合权重校准，测试数据仅用于评估。协议保持 exact path、source_id 和 MD5 disjointness，并记录 manifest SHA、checkpoint/prototype/calibration SHA provenance。

clean closed-set manifest 使用的是第三方或公共猪声音频来源，不包含本文新采集的动物录音。`dry_cough` 和 `abdominal_cough` 来自 Smart Farm Korea / data.go.kr 猪咳嗽声音数据集 [@smartfarmkorea_pig_cough_voice]。`calm_grunt`、`feeding`、`frightened_stress` 和 `anxious_stress` 来自 figshare Sow Call Dataset [@sow_call_dataset_2021]，本地子类型由原始文件名后缀映射而来。未解决许可的 Kaggle-like scream/cough 来源没有进入当前 cap3x 主线 manifest。

主文表格承担不同证据角色。Table 1 定义类别、子类型、backbone protocol、prototype protocol、noise protocol、leakage controls 和未完成外部验证。Table 2 和 Table 3 支撑 2 秒上下文故事。Table 4 和 clean prototype paired stats 支撑层级辅助监督故事。Table 5 支撑 clean 条件下 hierarchical prototype 的语义作用。Table 6 至 Table 9 支撑 simulated DEMAND noise、paired statistics、per-class failure modes 和 corrected selective prediction 的边界结论。

## Results

### Story 1：为什么是 2 秒上下文

问题是：猪声事件能否用 1 秒片段稳定表示。1 秒 Log-Mel CRNN 的 Macro-F1 为 0.9226，说明基础模型可用，但并未充分捕获事件上下文。随后尝试的方向包括更复杂的 frontend 和 architecture，例如 MCTAFD、PCEN、spectral-gate denoising、SpecAugment 和 attention pooling。失败证据是，这些改动没有稳定超过 2 秒 Log-Mel CRNN。对应均值分别低于 0.9472 的 2 秒主线。

因此主线收束为 duration-aware context。2 秒 Log-Mel CRNN 相对 1 秒的配对平均增量为 0.0246，Wilcoxon p=0.000162，成为 clean paper-main 方向。3 秒 Log-Mel 均值为 0.9434，未超过 2 秒。这说明当前任务需要足够事件上下文，但不是越长越好。结果边界是：2 秒只是在当前闭集协议和结果包中最受支持，不是所有猪声任务的普遍最优时长。

### Story 2：为什么引入层级辅助监督

问题是四个主类标签过粗。cough 内部存在 dry_cough 和 abdominal_cough，stress_vocal 内部存在 frightened_stress 和 anxious_stress，feeding 与 stress_vocal 边界在 confusion 中持续困难。尝试方案是把四类主任务和六类子类型任务绑定，以 `main loss + lambda * aux loss` 训练。

结果显示，lambda=0.2、0.5 和 1.0 的 mean Macro-F1 分别为 0.9505、0.9510 和 0.9515。lambda=1.0 是最高均值设置，lambda=0.5 是最低标准差设置。失败证据同样必须写清楚：层级辅助监督相对 2 秒基线没有达到统计显著。因此最终机制不是强性能结论，而是“层级监督提供细粒度语义和边界解释”。clean prototype 结果也支持这个边界：hierarchical prototype Macro-F1 为 0.9540，高于 raw Softmax 的 0.9510，但 hierarchical - raw_softmax 的 paired p=0.058253，仍不能写成显著主性能结论。

### Story 3：为什么做后验原型推理

问题是 raw Softmax 的判别边界在加性噪声下可能漂移。prototype 使用 train embedding 类中心表示类别声学原型，提供另一种冻结推理规则。main prototype 只使用四个主类中心；hierarchical prototype 加入六个子类中心。

干净条件下，hierarchical prototype 最高，Macro-F1 为 0.9540，说明子类型中心有利于语义组织。但 DEMAND simulated additive noise 下，main prototype 更稳。ALL_NOISY 条件下，raw Softmax、main prototype 和 hierarchical prototype 的 Macro-F1 分别为 0.6173、0.6230 和 0.6200。main prototype 相对 raw Softmax 的平均增量为 0.005755，95% bootstrap CI [0.001606, 0.010894]，Wilcoxon p=0.010511。hierarchical prototype 相对 raw Softmax 的平均增量为 0.002710，p=0.312333，未显著；hierarchical prototype 相对 main prototype 为 -0.003045，p=0.000376，说明噪声下层级原型显著弱于主类原型。

这个结果给出最终机制：层级原型适合 clean semantic explanation，主类原型适合 simulated-noise robustness。MODERATE_NOISE 中 main prototype 也优于 raw Softmax，0.6534 对 0.6472，paired p=0.001673；EXTREME_STRESS 中 main prototype 数值上高于 raw Softmax，0.5624 对 0.5574，但 p=0.560171，不能写成有统计支持的增益。

类别层面暴露了关键失败。ALL_NOISY 中 cough F1 很低：raw Softmax 为 0.0867，main prototype 为 0.0974，hierarchical prototype 为 0.0957。0 dB active-event SNR 下，STRAFFIC 和 TBUS 的 cough recall 对所有方法均为 0，DWASHING 中也低于 0.009。因此 0 dB 应写成 extreme stress condition，不能写成无噪声或可部署性能。

选择性预测也限制了 prototype 的表述。ALL_NOISY 中 raw Softmax 的 AURC/AUGRC 为 0.1775/0.1239，优于 main prototype 的 0.2109/0.1394 和 hierarchical prototype 的 0.2122/0.1403。prototype 只在 frozen-threshold selective risk 上略有改善，例如 ALL_NOISY 中 main prototype 为 0.2988，raw Softmax 为 0.3024。由此，prototype 不能被写成全面不确定性改进方法。fused 结果只作为消融，不是主创新点。

## Discussion

本文的核心不是提出一个更复杂的模型，而是把猪声识别的工程路线逐步收束。第一步收束到事件上下文：2 秒输入显著优于 1 秒，且 3 秒和复杂 frontend 没有提供更强 clean 主线。第二步收束到语义监督：层级辅助任务让表示包含子类型结构，但由于增益未显著，只能作为语义和边界贡献。第三步收束到噪声推理：clean 条件下层级原型解释性更强，噪声条件下主类原型更稳。

这种差异并不矛盾。子类型中心在 clean embedding 中可以提供更细粒度组织，但加性噪声可能让样本偏离具体子类型中心。主类中心更粗，因此在模拟噪声中更像稳定吸引子。也正因为如此，本文不应把 hierarchical prototype 写成最鲁棒方法，而应把 main prototype 写成中等噪声下更稳健的后验推理规则。

不确定性结果进一步约束了结论。若 prototype 同时改善 Macro-F1、calibration、AURC 和 AUGRC，才可以写成全面可靠性改进。当前结果不支持这种说法。raw Softmax 在 AURC/AUGRC 上排名更好，prototype 的优势主要体现在 aggregate Macro-F1 和 frozen-threshold selective risk 的小幅改善。

## Limitations

第一，DEMAND 是 simulated additive noise，不是真实猪场外部验证。第二，0 dB 是 active-event SNR 的极端压力条件，不是 no noise，也不是普通部署条件。第三，0 dB 下 cough 不可靠，STRAFFIC 和 TBUS 中所有方法 cough recall 均为 0。第四，层级辅助监督和 clean hierarchical prototype 都不能写成统计显著主性能提升。第五，prototype 没有全面改善不确定性，AURC/AUGRC 下 raw Softmax 排名更好。第六，fused 只是消融，不是主创新点。第七，原始猪声数据的伦理审批、共享权限、目标期刊格式和完整猪声文献引用仍需作者补充。

## Conclusion

本文支持三点有边界的结论。首先，2 秒 Log-Mel CRNN 是当前 clean closed-set 结果包中最受支持的主线，并相对 1 秒输入取得显著配对增益。其次，层级辅助监督提供子类型语义和边界解释，但不能被写成显著性能提升。最后，后验原型推理在干净和噪声条件下呈现不同作用：hierarchical prototype 更适合 clean 语义解释，main prototype 更适合 DEMAND simulated noise 下的鲁棒推理。未来工作应优先补充真实猪场外部验证、0 dB 强噪声 cough 识别、open-set/unknown 类处理和可公开复现的数据许可说明。

## Data Availability

本文草稿基于 `paper/tables/` 和 `paper/appendix/` 下的聚合 CSV 文件。代码、命令、聚合表格、图数据来源映射、交叉验证 manifest、hash-based leakage audit 和 provenance 记录可在仓库投稿包中提供。原始第三方猪声音频不随仓库再分发；韩国咳嗽录音应从 Smart Farm Korea / data.go.kr 获取 [@smartfarmkorea_pig_cough_voice]，Sow Call Dataset 录音应从 figshare 获取 [@sow_call_dataset_2021]。DEMAND 环境噪声是 Zenodo 上的公开数据 [@demand2018]，但也应通过原始记录获取，不应直接打包进仓库。未解决许可的 Kaggle-like scream/cough 来源不进入当前主结果，除非其 licence 和 owner rights 后续得到确认。

## Code Availability

当前文件中尚未提供投稿级代码可用性声明。正式投稿前应补充公开仓库地址、release tag 或归档 DOI。除非作者已经完成发布，否则本文不能声称代码、checkpoint 或原始数据已经公开。

## Ethics Statement

本文没有开展新的动物实验、动物干预或前瞻性猪场录音采集。当前工作复用公共或第三方猪声音频，并使用公开 DEMAND 环境噪声做计算分析。原始数据来源的伦理审批、权限和 repository terms 应归属于原始提供方，并在可获得时注明。正式投稿前，仍应在 reproducibility package 中保留未解决的来源权限项目，包括已排除的 Kaggle-like scream/cough 来源，以及韩国原始音频是否可公开再分发的最终条款复核。

## References placeholder

当前 `paper/references/references.bib` 中已有 DEMAND、prototypical networks、CRNN、selective classification、librosa 和 fd-shifts risk-coverage 实现等条目。投稿前必须补充并核验猪声识别、畜牧声学监测和农场噪声相关文献；不要虚构 DOI、作者、期刊或数据源。
