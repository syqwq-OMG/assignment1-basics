# Stanford leaderboard 对照：2026-10-20 15:25–15:30 +08:00

## 当前服务器结果

| 实验 | 最后 step | 完整验证 NLL | 完整验证 PPL | 总墙钟秒 |
| --- | ---: | ---: | ---: | ---: |
| 4L / d512 / batch16 / lr0.0012 | 45099 | 4.3674 | 78.8368 | 3755.5 |
| 8L / d512 / batch16 / lr0.0006 | 27770 | 4.2586 | 70.7135 | 3854.9 |

第三组 8L / d512 / batch16 / lr0.0012 尚在运行；step 24000 的抽样验证 NLL=4.3370（PPL 约 76.48），不是完整评分。当前证据支持较小学习率，但不能用抽样结果替代最终验证排名。

完整结果在服务器 `assignment1-formal-20261020/outputs/formal-search/experiments.json`。本次查看未修改运行源码和自动队列。

## 一手参考

- [Stanford 官方榜单](https://github.com/stanford-cs336/assignment1-basics-leaderboard)：当前 2026 规则为单 B200 45 分钟，2025 表为 H100 90 分钟；指标是 per-token validation loss，context=512。与我们的 A100 六小时不能按时间直接换算，且不同学生 BPE 会影响 token-level 指标。
- [Jiaxin Fang，PR #124](https://github.com/stanford-cs336/assignment1-basics-leaderboard/pull/124)：提交报告 loss=3.32098；AdamW，8L、d_model=768、d_ff=2048、12 heads、训练 context=1024、batch=128；LR 0.003→0.0003，warmup 500，total 12000，WD=0.1，betas=(0.9,0.95)，clip=1；BF16 autocast 和 torch.compile。这是提交者报告，不能据此保证在本次环境复现同等分数。
- [Gorn Namchittai，PR #156](https://github.com/stanford-cs336/assignment1-basics-leaderboard/pull/156)：提交报告 loss=3.4348；8L、d768、FFN2048、12 heads、batch128；LR 0.005→0.0005，按墙钟调度，warmup300秒，WD0.15；TF32 和 torch.compile。
- [Thomas Li，PR #160](https://github.com/stanford-cs336/assignment1-basics-leaderboard/pull/160)：提交报告 loss=3.03543；16L、d1024、FFN4096、8 heads、batch256；NorMuon+AdamW、BF16、整体编译、80% linear cooldown；另有多项架构改动，不适合作为最小改动方案直接搬用。
- [Herman Brunborg，PR #70](https://github.com/stanford-cs336/assignment1-basics-leaderboard/pull/70)：16L、d1024、8 heads、context512、batch128、约20k updates；Muon+AdamW、BF16、compile、U-Net skip。作者报告解绑输入/输出权重反而更好，故不能将 weight tying 当成普遍必然的收益。
- [Brandon Snider 最终配置](https://github.com/brandon-snider/cs336-a1/blob/main/leaderboard/configs/on-6.yml)及[基础配置](https://github.com/brandon-snider/cs336-a1/blob/main/leaderboard/configs/base.yml)：12L、d1280、FFN5120、batch128、context512，Muon、ReLU²、weight tying、额外 residual，LR0.0016线性降至0。查看的是公开配置，不导入其实现。

## 对本次实验的判断

1. 当前搜索固定 d512 和 batch16，只测深度和 LR，范围偏保守。值得补测 d768、8L、FFN2048、12 heads，batch32/64。batch 改变后应重新搜索 LR，不能直接沿用 batch16 的最优 LR，也不能把 Muon 的 LR 用到 AdamW。
2. 优先保留手写组件和 AdamW，对比 BF16 与 torch.compile 的实际 tokens/s、数值稳定性和完整验证正确性。编译时间计入六小时预算；涉及训练代码改动应另开 branch，并在独立目录测试。
3. 正式长跑仍应按实际时限完成 warmup/cosine cooldown。当前 pilot 的长周期几乎不衰减，是搜索阶段设置，不能据此判断最终低学习率阶段的效果。
4. 宽度、batch、精度和编译分别测试，避免同时修改后无法判断收益。BF16 应确保手写 softmax/CE/归一化等归约的数值稳定性，不能只机械添加 autocast。
5. 只保留通过完整验证、计时和源码记录核查的成绩。榜单低 loss 对应 PPL=exp(loss)，不是 PPL=loss；不同 tokenizer、验证覆盖、硬件及优化器影响直接可比性。

以上是调研与下一轮候选建议，尚未变更正在执行的五组搜索和两组正式长跑队列。
