# 《端侧大模型推理：原理与 LiteRT-LM 实现》

> *On-Device LLM Inference: Principles and Practice with LiteRT-LM*
> **内部打磨版 · 私有仓库 · 非官方**。开源与否未决策前，勿向外部渠道发布任何内容片段。

一本讲透"如何让大语言模型在手机、手表、浏览器等受限设备上高效运行"的书，
以 Google 投产的端侧 LLM 运行时 [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) 为解剖标本。

## 先读这两份

- **`BOOK_PLAN.md`** — 全书规划：立意、目录（11 章 + 尾声）、章卡、路线图、风险登记册。
- **`CLAUDE.md`** — 写作规范：每个写作会话逐字加载，含断言四级制、除 AI 味清单、图表规范。

## 目录结构

```
chapters/chNN-slug/      每章：chapter.md（正文）· notes.md（素材）· review.md（两个 pass 留痕）· figs/
appendix/                附录 A 术语表 · B 代码地图 · C 复现指南 · D 基准数据集
assets/book.css          SVG 配图的 CSS 变量表（单一事实源）
scripts/lint_prose.sh    文风机检（禁词/填充词/中英文空格）
experiments/             实验脚本与基准数据
SUMMARY.md book.toml     mdBook 配置
```

## 构建（需要时）

```bash
# 安装 mdBook（Rust 工具链）
cargo install mdbook
# 本地预览
mdbook serve --open
```

## 代码锚点

全书代码引用锁定 `google-ai-edge/LiteRT-LM @ v0.13.1`。核对代码时在**独立 worktree** checkout 该 tag，
不扰动 `/Users/dpthinker/workspace/LiteRT-LM` 的 main 工作区。
