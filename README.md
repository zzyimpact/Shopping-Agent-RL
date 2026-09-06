# Shopping Agent RL

## A ShopSimulator RL Post-Training Reproduction

本项目基于公开的 ShopSimulator 任务、商品目录和购物环境，复现面向 Qwen3-8B 的单轮购物 agent post-training 闭环：teacher rollout、成功轨迹筛选、scenario-specific SFT、在线 GRPO 和 held-out evaluation。v1 只覆盖 `single` 与 `single_persona` 两个 scenario，并优先复用相邻的 ShopSimulator checkout；本阶段不实现训练、teacher collection 或正式评测。

项目定义见 [DESIGN.md](DESIGN.md)，实施顺序与验收边界见 [WORKFLOW.md](WORKFLOW.md)。

当前状态：P0 upstream audit 已完成；P1 Remote CPU Bootstrap 已完成基础依赖与脚本准备，等待解决远程 checkout 定位后进入 P2。
