# 最终图件 figure-to-claim 审计

## 审计模式与结论

- 审计日期：2026-07-11
- 路线：academic-research-suite 的 integrity verification、claim alignment 与 methodology reviewer 只读路线
- 审计对象：5 张主图、3 张补充图、双语图注、source map、figure-to-claim matrix 与 QA 报告
- 最终结论：**PASS WITH NOTES**
- 阻断问题：0
- 严重或中等完整性问题：0
- 禁止性过度声明：0
- 面板覆盖：25/25

> This check verifies disclosure and claim-to-provenance fidelity. It does not judge whether the experiment was correctly designed, run, statistically adequate, or reproducible by ARS.

本审计核对的是图件陈述是否忠实于锁定结果及其证据边界，不重新评价实验设计，也不运行模型。图件包没有新增文献型引用，因此 citation-integrity 路线在本轮用于核对内部来源、数值、图注和主张的一致性；未新增、删除或改写参考文献。

## 只读边界

- 未运行训练、推理、特征提取、音频处理或 checkpoint 加载。
- 未修改 manuscript、references、模型代码、音频或任何原始结果 CSV/JSON。
- 生成器仅读取锁定的 paper-ready CSV/Markdown 与一个协议映射模块，并仅写入批准的图件、source-data 和 review 目录。
- 39 个任务开始前锁定的结果 CSV/JSON 在最终复核中保持相同 SHA-256。
- 生成器还在每次导出前后比较全部必需证据源哈希；最终导出未检测到来源变化。

## 证据包

- 面板来源：`paper/figure_source_data/FIGURE_SOURCE_MAP.csv`
- 图件主张：`paper/figure_source_data/FIGURE_TO_CLAIM_MATRIX.csv`
- 双语图注：`paper/figures_journal/FIGURE_LEGENDS_CN.md`、`FIGURE_LEGENDS_EN.md`
- 导出与视觉 QA：`paper/figures_journal/FIGURE_QA_REPORT.md`
- 主要锁定结果：`paper/tables/*.csv`、`paper/appendix/*.csv`、`paper_results/tables/*.csv`
- 同队列直接证据：`reports/prototype_noise_demand_w05_final/final_runs.csv`
- 混淆矩阵直接证据：`reports/prototype_noise_demand_w05_final/final_confusion_summary.csv`

## 25 个面板的 trace 与主张审计

| 图件 | 面板 | 主要证据 | 核对结果 | 结论边界 |
|---|---|---|---|---|
| Figure 1 | a | table1；只读 manuscript 数据来源段 | PASS | 公共/第三方音频的计算复用；不声称新采集流程 |
| Figure 1 | b | table1；`prototype_model_adapter.py` 映射 | PASS | 四主类、六子类型；不存在自动路由 |
| Figure 1 | c | table1 协议 | PASS | prototypes 仅由训练嵌入构建，校准仅用验证集 |
| Figure 1 | d | table1 协议 | PASS | DEMAND 为模拟加性噪声，不是真实猪场外部验证 |
| Figure 2 | a | table3 | PASS | 1/2/3 s 为描述性均值 ± 样本 SD；n=25/25/15 |
| Figure 2 | b | duration paired stats | PASS | 仅比较匹配的 25 个 2 s−1 s fold-seed 对 |
| Figure 2 | c | table2 | PASS | n=15 与 n=25 行不伪装成配对比较 |
| Figure 3 | a | table1；subtype 映射 | PASS | 标签层级是监督结构，不外推生物学层级 |
| Figure 3 | b | table4；三个 paired-stats 文件 | PASS | lambda=1.0 均值最高、lambda=0.5 SD 最低；无显著增量声称 |
| Figure 3 | c | table5；clean paired stats；`final_runs.csv` | PASS | 25 个唯一 fold-seed 对均为同一 lambda=0.5 checkpoint cohort |
| Figure 4 | a | table6 | PASS | 三个分层中 main prototype 描述性均值最高，增量较小 |
| Figure 4 | b | noise summary | PASS | 仅使用 clean 参考及三个 DEMAND 环境的 20/10/0 dB 行，排除 GROUP 汇总行 |
| Figure 4 | c | run-level paired stats；fold-cluster bootstrap | PASS | 两层区间与 P 值分开；P 值按源文件原样报告且未作多重性校正 |
| Figure 5 | a | per-class summary | PASS | severe simulated noise 下 cough recall 崩塌；0 dB 不是无噪声 |
| Figure 5 | b | table8 | PASS | pooled repeated decisions 不是独立录音，不作因果解释 |
| Figure 5 | c | AURC/AUGRC；selective summary | PASS | lower is better；raw Softmax 风险排序更好，prototype 仅有小的固定点优势 |
| Supplementary Figure S1 | a | table2 全部 9 行 | PASS | 未加入未锁定的 BiLSTM 或归档值 |
| Supplementary Figure S2 | a | noise summary，raw Softmax | PASS | clean + 3 environments × 3 SNR，排除 GROUP 行 |
| Supplementary Figure S2 | b | noise summary，main prototype | PASS | 同上；不声称逐环境显著 |
| Supplementary Figure S2 | c | noise summary，hierarchical prototype | PASS | 同上；不声称 hierarchical 在噪声下占优 |
| Supplementary Figure S2 | d | noise summary，fused | PASS | fused 仅为补充消融，不作为主贡献 |
| Supplementary Figure S3 | a | final confusion summary | PASS | 四个完整 4×4 矩阵；每个真实类行汇总 9,450 次重复决策 |
| Supplementary Figure S3 | b | calibration distribution | PASS | 25 次验证集校准的完整离散选择频率 |
| Supplementary Figure S3 | c | calibration distribution | PASS | 25 次验证校准的全局/逐类冻结阈值分布 |
| Supplementary Figure S3 | d | fold-level deltas | PASS | 三个分层 × 三种比较 × folds 0–4；fold 点不等同于 25 个独立 runs |

## 定量复核

### 时长主结果

- 1 s：Macro-F1 0.9226，n=25。
- 2 s：Macro-F1 0.9472，n=25。
- 3 s：Macro-F1 0.9434，n=15；未与 2 s 构造未锁定的配对检验。
- 匹配的 2 s−1 s 平均差为 0.024630891。
- 10,000 次配对差值 bootstrap 95% CI 为 [0.014112838, 0.035642072]。
- 双侧 Wilcoxon P=0.0001623034477；图中按有效位数显示准确数值，不使用显著性星号。

### 层级监督与干净原型

- lambda=0.2、0.5、1.0 的 paired Wilcoxon P 分别为 0.265132、0.170241、0.153127，三个 CI 均跨 0。
- lambda=1.0 的均值最高；lambda=0.5 的跨运行 SD 最低。
- `final_runs.csv` 含 25 个唯一 fold-seed 对，全部 lambda=0.5；由此直接支持 Figure 3c 的同队列限定。
- clean main prototype−raw、hierarchical−raw、hierarchical−main 的配对 CI 均跨 0，P 均大于 0.05；图注未声称显著提升。

### 模拟噪声

- main prototype−raw 在 moderate、ALL_NOISY、extreme 的 run/fold P 分别为：
  - 0.00167307 / 0.0625；
  - 0.0105108 / 0.3125；
  - 0.560171 / 1.0。
- 图件把 25-pair run-level 与 5-fold cluster sensitivity 明确分开；没有把 run-level 结果单独包装为跨 fold 的稳健显著性。
- 九组 run/fold P 为源文件中保存的未校正 P 值；图注和 source map 已明确这一点。
- moderate-noise 证据最强；0 dB 只被描述为极端模拟噪声压力条件；main prototype 而非 hierarchical prototype 是噪声下最稳健的已支持选择。

### 失败模式与选择性预测

- feeding→stress / stress→feeding 的 pooled counts：raw 2168/616，main prototype 2006/606，hierarchical 2240/495，与 table8 一致。
- Raw Softmax 的 ALL_NOISY AURC/AUGRC 最低；main prototype 仅在 frozen-threshold selective risk 上略低。
- 选择性指标没有锁定的配对 CI 或 P 值，图中没有虚构误差线或显著性。

## 视觉与统计披露审计

- 所有 Macro-F1/recall 主轴均为完整 0–1；仅明确标为差值的面板围绕 0 使用紧凑轴。
- Figure 2 panel-c 数值标签位于 SD 区间之后，不再压住误差线。
- Figure 3 panel-b 在图内直接写明“无显著增量干净收益”。
- Figure 4 panel-a 在图内突出 main prototype 与 moderate-noise 证据边界。
- Figure 4 的 run-level 与 fold-cluster marker、区间及 exact P 分开显示；无显著性星号。
- Figure 4 最终 PNG/TIFF 右内容边距分别为 10 px/21 px，最后一个差值轴标题未裁切。
- 图例不遮挡数据；8 张 PNG 均按原始分辨率完成手工视觉检查。
- 主文图恰为 5 张，补充图恰为 3 张；详细负结果未从补充图中删除。

## 禁止性过度声明检查

未检测到以下禁止性表述：

- 把 DEMAND 模拟加性噪声写成真实猪场外部验证；
- 把 0 dB 写成 no-noise；
- 把层级监督或干净 hierarchical prototype 的小增量写成显著；
- 把 Figure 3c 的 raw Softmax 当作非层级 2 s baseline；
- 把 pooled repeated decisions 当作独立录音；
- 声称 prototype 普遍改善不确定性；
- 声称 hierarchical prototype 在噪声下最稳健；
- 把 n=15 与 n=25 的未匹配行当作严格配对检验。

## 已解决的审计发现

1. Figure 1a 已加入公共/第三方音频陈述的直接来源。
2. Figure 3c 已加入 `final_runs.csv`，直接锁定同一 lambda=0.5 队列。
3. Supplementary Figure S2 已明确排除 GROUP 汇总行；Figure 4b/5a/5c 的过滤也同步精确化。
4. Figure 1 双语图注已明确 n 不适用。
5. Figure 4 图注/source map 已明确 P 值未作多重性校正。
6. Figure 5 已明确 n 是 fold-seed run，panels a/b 使用同一批 runs 的 pooled repeated decisions。
7. Figure 4 右侧触边已修复并通过像素边距复核。
8. 独立代码复核提出的 Figure 2 标签压线、Figure 3 非显著标注和 Figure 1 原型分支歧义均已修复。

## 保留说明

- 25 个 fold-seed runs 不是 25 个独立生物样本；5-fold cluster 检验只有 5 个聚类，功效低。
- 没有锁定的逐 environment×SNR fold-cluster CI，也没有选择性指标的配对 CI/P。
- DEMAND 控制性加性噪声不等于真实猪场外部验证。
- SimSun 在 SVG 中保留为可编辑 text nodes，但未嵌入；缺少 SimSun 的系统可能发生字体替代与重排。
- 本地没有外部 PDF 字体检查程序；PDF 通过成功导出、签名和非空检查，最终投稿前仍建议用期刊生产环境再次 preflight。

## 最终判定

图件包可用于当前中文 Word 工作稿和后续期刊排版：**PASS WITH NOTES**。Notes 均已在图注、QA 或本报告中显式披露，不构成图件包阻断项。未要求、也未执行论文正文改写。
