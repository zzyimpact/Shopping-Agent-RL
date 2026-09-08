# Shopping Agent RL

> **Work in Progress / 项目仍在开发中**

当前状态：P2、P3-0、P3a、P3b 已完成；P3c formal collector implementation 已准备好，
等待用户执行 tiny paid formal smoke。项目尚未完成最终
teacher 数据集、训练或实验结果，本文不宣称最终性能。

## A ShopSimulator RL Post-Training Reproduction

本项目基于公开的 ShopSimulator 任务、商品目录和购物环境，复现面向 Qwen3-8B 的单轮购物 agent post-training 闭环：teacher rollout、成功轨迹筛选、scenario-specific SFT、在线 GRPO 和 held-out evaluation。v1 只覆盖 `single` 与 `single_persona` 两个 scenario，并优先复用相邻的 ShopSimulator checkout；本阶段不实现训练、teacher collection 或正式评测。

项目定义见 [DESIGN.md](DESIGN.md)，实施顺序与验收边界见 [WORKFLOW.md](WORKFLOW.md)。

当前状态：P0/P1/P2 已完成；P3 teacher-data pipeline in progress。P3c 尚待 tiny smoke
验收，训练、正式 teacher
collection 与最终评测仍未开始。
