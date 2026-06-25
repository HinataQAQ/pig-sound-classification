# 面向模拟环境噪声鲁棒猪声识别的时长感知层级表征与双原型推理方法

摘要：面向猪场环境中的猪声识别，本文在已验证的 class-capped cap3x、2 秒 Log-Mel CRNN 主线基础上，研究时长感知层级表征、后验声学原型推理以及模拟环境噪声下的鲁棒性。干净语音闭集实验表明，2 秒上下文相对 1 秒输入带来显著的 Macro-F1 提升；层级辅助子类监督引入了细粒度声学语义并改善 feeding 与 stress_vocal 边界，但其相对 2 秒基线的增益未达到统计显著。在 lambda=0.5 的 25 个 fold-seed clean 原型实验中，层级原型 Macro-F1 为 0.954。在 DEMAND DWASHING、TBUS 与 STRAFFIC 的零样本冻结模拟噪声测试中，main prototype 在 ALL_NOISY 条件下 Macro-F1 为 0.623，较 raw Softmax 的 0.6173 平均提升 0.005755，配对 Wilcoxon p=0.010511。结果说明，未经噪声重训练的主类原型推理可提升模拟环境噪声下的整体鲁棒性，但本文未进行真实猪场外部验证，也不声称 0 dB 咳嗽识别已经可靠。

## 关键词

猪声识别；Log-Mel；CRNN；层级监督；原型推理；模拟环境噪声；选择性分类

## 1 引言

猪声自动识别可为规模化养殖中的健康和行为监测提供工程化感知入口。本文关注四类主声学事件：cough、calm_grunt、feeding 和 stress_vocal。与真实猪场部署相比，当前研究只覆盖干净数据和 DEMAND 环境噪声模拟，因此结论严格限定为闭集分类与模拟噪声鲁棒性。

## 2 方法

主模型采用 2 秒 Log-Mel 输入和 CRNN backbone，并在主类分类之外加入六个辅助子类监督。后验原型模块冻结 Softmax 主模型，用每个 fold-seed 的 train split 构建主类和辅助子类 embedding 原型，用 validation split 校准温度、阈值和融合权重，并在 test split 上冻结评价。

## 3 实验协议

所有实验保持 path、source_id 和 MD5 去重边界。干净模型比较采用匹配 fold-seed 配对统计。模拟噪声实验使用 DEMAND 的 DWASHING、TBUS 和 STRAFFIC，活动区 SNR 为 20、10 和 0 dB。噪声绘制、SNR、threshold 和统计协议在 final 25-run 前冻结，测试集不参与任何调参。

## 4 结果

2 秒上下文是干净数据上的主要显著增益来源。层级辅助监督的平均 Macro-F1 高于 2 秒基线，但相对增益不显著，因此只作为细粒度语义与边界改善证据。clean 原型实验显示 hierarchical prototype 的 Macro-F1 为 0.954。

在模拟噪声 ALL_NOISY 聚合条件下，raw Softmax Macro-F1 为 0.6173，main prototype 为 0.623，hierarchical prototype 为 0.62。main prototype 相对 raw Softmax 的平均增益为 0.005755，95% bootstrap CI 为 [0.001606, 0.010894]，Wilcoxon p=0.010511。hierarchical prototype 相对 raw Softmax 的平均增益为 0.00271，未达到统计显著。

## 5 讨论与局限

本文不能声称层级原型是最鲁棒方法，也不能声称完成真实猪场外部验证。0 dB 活动区 SNR 下 cough recall 明显退化，说明强噪声下咳嗽识别仍是主要风险。AUGRC 已按 fd-shifts generalized risk 定义修正，使用已保存 prediction CSV 重新计算。

## 6 结论

结果支持三点：2 秒上下文显著改善干净分类；层级辅助监督提供细粒度语义但增益不显著；主类原型推理在模拟 DEMAND 噪声下显著提升 aggregate Macro-F1，且不需要噪声重训练。
