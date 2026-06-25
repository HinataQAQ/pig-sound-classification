# GPT -> Codex Handoff

APPROVED: true

## Approved Task
待 GPT Pro 填写。Codex 只执行明确标记 `APPROVED: true` 的任务。

## Required End-of-Round Instruction

本轮任务完成后，不要自动开始下一阶段。

请更新：

- handoff/CODEX_TO_GPT.md
- handoff/CODEX_TO_GPT.json
- handoff/HISTORY.md

内容必须包含：

1. 实际执行的命令；
2. branch 和 commit；
3. 新增/修改文件；
4. 所有核心指标；
5. train/val/test 使用边界；
6. path/source_id/MD5 泄漏审计；
7. paper_usable=true/false；
8. 与已验证 baseline 的比较；
9. 失败项；
10. 需要 GPT Pro 判断的问题；
11. 建议下一步，但不得自动执行。

完成后 commit 和 push，然后停止。
