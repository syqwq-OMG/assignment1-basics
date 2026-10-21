# 最终结果与提交

## 提交内容

按 `ignored/Assignment/Assignment_1_Basic.md`，在 [Assignment1 表单](https://f.wps.cn/g/taSFb1GF/) 填写姓名、学号、BPE 和 Transformer 两个正式 W&B run 链接；存在续训时附关联run。提交前确认助教能够访问链接。

建议使用：

- BPE：[507bb0852a58](https://wandb.ai/syqwq-east-china-normal-university/lmfs-assignment1/runs/507bb0852a58)。32000词表，全文还原通过，完整验证耗时291.686秒。
- Transformer：[d8a4c914791a](https://wandb.ai/syqwq-east-china-normal-university/lmfs-assignment1/runs/d8a4c914791a)。完整验证 **PPL 27.938069335964723**。

W&B需保留源码及快照、BPE前200个token、完整验证耗时、Transformer训练PPL曲线、最后模型的完整验证PPL、最终步数和累计耗时。文档未要求额外报告或代码zip；模型与分词文件须自行保留供复查。

## 两组正式结果

| 配置 | 最后更新步数 | 完整验证PPL | 累计秒 | 预算内 |
| --- | ---: | ---: | ---: | --- |
| 8层、宽度768、batch64、BF16+compile | 96317 | **27.9381** | 20853.8795 | 是 |
| 8层、宽度768、batch32、BF16+compile | 178177 | 30.1412 | 20856.4280 | 是 |

两组均覆盖66401097个验证目标，采用完整验证总NLL除以有效目标数再取exp。W&B状态均为finished，`source_changed=[]`，GPU任务已于2026-10-21 07:37结束。

对照组：[14d3831732c9](https://wandb.ai/syqwq-east-china-normal-university/lmfs-assignment1/runs/14d3831732c9)。

## 服务器文件

最佳模型：

```text
/ai/lmfs2026/sunyuquan/assignment1-recovery-20261020/outputs/d8a4c914791a/last.pt
/ai/lmfs2026/sunyuquan/assignment1-recovery-20261020/outputs/d8a4c914791a/result.json
```

正式BPE前缀：

```text
/ai/lmfs2026/sunyuquan/assignment1-basics/outputs/507bb0852a58/bpe/tokenizer
```

调度结果和全部排名：

```text
/ai/lmfs2026/sunyuquan/assignment1-recovery-20261020/outputs/recovery/experiments.json
/ai/lmfs2026/sunyuquan/assignment1-recovery-20261020/outputs/recovery/ranking.json
```

本仓库保存最佳超参数于 [configs/owt_best.json](../configs/owt_best.json)。已完成实验对应的服务器文件和源码快照无需因目录整理而移动。
