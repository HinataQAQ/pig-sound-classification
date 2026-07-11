# Pig Sound Classification Nomenclature v1

本文档定义活跃代码、新日志、新图表和新输出所使用的科学命名。历史
artifact、checkpoint、CSV/JSON schema 与文件名保持原样；读取端通过
`tools/nomenclature.py` 解释旧名称。

当前 schema 版本为 `pig_sound_nomenclature.v1`。

## 1. Architecture 与 inference route 是两个维度

模型架构描述特征输入、主干网络及训练监督；inference route 描述同一冻结模型
输出如何形成分类或候选结果。二者不得混用。

- 主干模型是 validation-selected 2-s hierarchical-supervision CRNN。
- 主分类器是 **Primary Softmax route (Raw Softmax)**，直接使用主分类头输出。
- **Main-class prototype candidate route** 与
  **Hierarchical prototype candidate route** 是并行候选路由，不替代主分类器。
- 只有把 hierarchical-prototype 的 Top-1 作为最终决策时，才属于 B3 消融。
  “hierarchical supervision” 本身不等于 B3。

`calibrated_softmax`、CLI 别名 `softmax` 和 `fused` 仍可被旧流程读取，但它们不属于
本版三个 canonical inference routes，不能强行映射为其中之一。

## 2. B0/B1/B2/B3

| Stage | Canonical ID | English display name | 中文显示名 | 定义 |
|---|---|---|---|---|
| B0 | `b0_1s_logmel_baseline` | B0 — 1-s Log-Mel baseline | B0 — 1 秒 Log-Mel 基线 | 1 秒 Log-Mel CRNN 基线。 |
| B1 | `b1_2s_logmel_mainline` | B1 — 2-s Log-Mel mainline | B1 — 2 秒 Log-Mel 主线 | 2 秒上下文 Log-Mel CRNN 主线。 |
| B2 | `b2_validation_selected_hierarchical_crnn` | B2 — Validation-selected 2-s hierarchical-supervision CRNN | B2 — 验证集逐折选择的 2 秒层级辅助监督 CRNN | 每折只用验证集选择 λ 的 2 秒层级辅助监督 CRNN；Primary Softmax 是主分类路由。 |
| B3 | `b3_hierarchical_prototype_top1_ablation` | B3 — Hierarchical-prototype Top-1 decision ablation | B3 — 层级原型 Top-1 决策消融 | 将 hierarchical-prototype Top-1 用作决策的消融；不得表述为已优于 B2。 |

## 3. Canonical IDs 与双语标签

### Model family

| Canonical ID | English | 中文 |
|---|---|---|
| `logmel_crnn` | Log-Mel CRNN | Log-Mel CRNN |
| `hierarchical_supervision_crnn` | Hierarchical-supervision CRNN | 层级辅助监督 CRNN |

### Inference route

| Canonical ID | English | 中文 |
|---|---|---|
| `primary_softmax` | Primary Softmax route (Raw Softmax) | 主 Softmax 路由（原始 Softmax） |
| `main_class_prototype_candidate` | Main-class prototype candidate route | 主类别原型候选路由 |
| `hierarchical_prototype_candidate` | Hierarchical prototype candidate route | 层级原型候选路由 |

### Selection protocol

| Canonical ID | English | 中文 |
|---|---|---|
| `none` | No selection protocol recorded | 未记录选择协议 |
| `foldwise_validation_selected_lambda` | Fold-wise validation-selected λ | 逐折验证集选择 λ |
| `fixed_lambda_0_5_retrospective` | Fixed λ=0.5 retrospective | 固定 λ=0.5 回顾性 |

## 4. Legacy aliases

旧命令、旧 JSON key、旧 CSV method 值继续有效。读取时按下表解析；历史文件本身
不改写。

| Legacy value | Canonical inference route |
|---|---|
| `raw_softmax` | `primary_softmax` |
| `prototype` | `main_class_prototype_candidate` |
| `main_prototype` | `main_class_prototype_candidate` |
| `hierarchical` | `hierarchical_prototype_candidate` |
| `hierarchical_prototype` | `hierarchical_prototype_candidate` |

其他 namespace 的兼容别名如下。Training-stage 别名只解析 stage 身份，绝不隐式
设置 selection protocol。

| Namespace | Legacy value(s) | Canonical ID |
|---|---|---|
| `model_family` | `hier_longcontext_crnn`, `hierarchical_crnn` | `hierarchical_supervision_crnn` |
| `training_stage` | `b0` | `b0_1s_logmel_baseline` |
| `training_stage` | `b1` | `b1_2s_logmel_mainline` |
| `training_stage` | `b2`, `b2_valsel` | `b2_validation_selected_hierarchical_crnn` |
| `training_stage` | `b3` | `b3_hierarchical_prototype_top1_ablation` |
| `selection_protocol` | `validation_selected`, `foldwise_validation_selected` | `foldwise_validation_selected_lambda` |
| `selection_protocol` | `fixed_lambda_0.5`, `fixed_lambda=0.5`, `fixed_lambda_0_5`, `fixed_w05` | `fixed_lambda_0_5_retrospective` |

新 CLI 应优先使用 canonical value。使用上述 legacy value 时可发出
`DeprecationWarning`，但不得失败。现有存储列名仍使用兼容 key，例如
`raw_softmax_prob_*`、`prototype_prob_*` 与 `hierarchical_prob_*`。

## 5. Fixed λ=0.5 与 validation-selected 严格分离

`fixed_lambda_0_5_retrospective` 是固定 λ=0.5 的回顾性分析；
`foldwise_validation_selected_lambda` 是每折只根据验证集选择 λ 的协议。即使某一折
最终选择值恰好为 0.5，也不能据此把 fixed-0.5 artifact 标记为 validation-selected，
反向映射同样禁止。

Canonical method ID 把 stage、selection protocol 和 inference route 同时编码：

```text
<training_stage>::<selection_protocol>::<inference_route>
```

例如：

```text
b2_validation_selected_hierarchical_crnn::foldwise_validation_selected_lambda::primary_softmax
b2_validation_selected_hierarchical_crnn::fixed_lambda_0_5_retrospective::primary_softmax
```

两者必须保持为不同 method ID。

读取端必须继承并校验已记录的 protocol。artifact 已记录具体 protocol 时，CLI
不得用另一个具体 protocol 覆盖；只有来源字段缺失或为 `none` 时，才可用显式 CLI
参数补充。多运行汇总必须先确认来源的 `model_family`、`context_seconds` 和
`selection_protocol` 一致，不得用默认值掩盖冲突。

## 6. 历史 artifacts 规则

以下内容只读并保持 byte-for-byte 不变：

- checkpoint 文件名、checkpoint bytes 与 `state_dict` keys；
- `main_head`、`aux_head` module attributes；
- 既有 `reports/`、`paper/`、`paper_results/` 输出；
- 既有 CSV/JSON 内容、schema column names、comparison strings；
- manifests、SHA/MD5 记录、已发布指标和图件；
- Git branches、tags 与历史 commits。

代码只能在读取边界解析 legacy value，在新输出中追加 metadata。不得为了“统一”而
对仓库执行全局替换。若 artifact 没有记录 selection protocol，安全默认值是
`none`；不得从目录名或 λ 数值猜测 validation-selected。

活跃生成器把本版新结果写入带版本标识的新位置，例如
`paper/final_validation_audit_nomenclature_v1/`、
`paper/figures_final_nomenclature_v1/`，以及带 `_nomenclature_v1` 后缀的累计分析
文件；既有无版本目录只作为冻结输入或历史交付物。目标已存在时必须拒绝覆盖。

## 7. 新 metadata

具有单一 inference route 的新输出应包含以下字段：

```json
{
  "nomenclature_schema_version": "pig_sound_nomenclature.v1",
  "model_family": "hierarchical_supervision_crnn",
  "training_stage": "b2_validation_selected_hierarchical_crnn",
  "context_seconds": 2.0,
  "selection_protocol": "foldwise_validation_selected_lambda",
  "inference_route": "primary_softmax",
  "canonical_method_id": "b2_validation_selected_hierarchical_crnn::foldwise_validation_selected_lambda::primary_softmax",
  "display_name_en": "Primary Softmax route (Raw Softmax)",
  "display_name_zh": "主 Softmax 路由（原始 Softmax）",
  "legacy_method_id": "raw_softmax"
}
```

同时包含多个 route 的 bundle 或 analysis-level provenance 没有唯一
`inference_route`，因此只写适用的公共字段，并可另列各 route 的完整 metadata；
不得填入一个虚假的单一路由。

## 8. Python 使用示例

```python
from tools.nomenclature import (
    build_method_metadata,
    canonicalize_inference_route,
)

route = canonicalize_inference_route("raw_softmax", warn_on_legacy=True)
metadata = build_method_metadata(
    model_family="hierarchical_supervision_crnn",
    training_stage="b2_validation_selected_hierarchical_crnn",
    context_seconds=2.0,
    selection_protocol="foldwise_validation_selected_lambda",
    inference_route=route,
    legacy_method_id="raw_softmax",
)
```

直接运行 `python tools/<script>.py` 的旧入口仍受支持；活跃脚本应使用 package/direct
execution 双兼容 import，或沿用其现有的 sibling-import 约定。
