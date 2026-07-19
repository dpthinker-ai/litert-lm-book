# 《端侧大模型推理：原理与 LiteRT-LM 实现》

> *On-Device LLM Inference: Principles and Practice with LiteRT-LM*
> **内部打磨版 · 私有仓库 · 非官方**。开源与否未决策前，勿向外部渠道发布任何内容片段。

本书分析大语言模型如何在手机、手表和浏览器等受限设备上运行，
以 Google 的端侧 LLM 运行时 [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) 为主要实现对象。

## 先读这两份

- **`BOOK_PLAN.md`** — 全书规划：立意、目录（11 章 + 尾声）、章卡、路线图、风险登记册。
- **`AGENTS.md`** — 写作规范：每个写作会话逐字加载，含断言四级制、除 AI 味清单、图表规范。

## 目录结构

```
parts/                    四个正文部分的导页
chapters/chNN-slug/      每章正文与 SVG 配图
archive/chapters/        素材与 review.md 验收记录
appendix/                附录 A-E
assets/book.css          SVG 配图的 CSS 变量表（单一事实源）
scripts/lint_prose.sh    文风机检（禁词/填充词/中英文空格）
scripts/check_citations.py  页下注完整性与所有附录来源去重检查
scripts/check_code_references.py  源码锚点文件与行号检查
scripts/check_book_links.py  成书 HTML 本地链接与锚点检查
scripts/build_book.sh    白名单 staging + mdBook 构建 + 泄漏检查
scripts/make_pdf.sh      WebKit 连续排版 + A4 分页 + 页下注 + 目录/书签/页脚
experiments/             实验脚本与基准数据
SUMMARY.md book.toml     mdBook 配置
```

## 构建（需要时）

```bash
# 安装 mdBook（Rust 工具链）
cargo install mdbook
# 安全构建：只把正文、附录、图和样式放进临时书稿源
bash scripts/build_book.sh

# 生成 dist/book.pdf；当前同页页下注排版需要 macOS WebKit
bash scripts/make_pdf.sh --debug

# 需要本地服务时，先刷新 staging，再启动 mdBook
bash scripts/stage_book.sh
mdbook serve --open
```

不要直接把仓库根目录作为 mdBook `src`。构建脚本会检查 `.git`、`archive/`、`experiments/`、`scripts/` 和 `dist/` 未进入 `book/`。
HTML 保留可点击的语义脚注。PDF 构建会把同一来源排在引用所在页底部，同页去重、跨页重印；非 macOS 环境检测到语义脚注时会中止，避免 Chromium 静默生成章末注。

## 代码锚点

全书代码引用锁定 `google-ai-edge/LiteRT-LM @ v0.13.1`。核对代码时在**独立 worktree** checkout 该 tag，
不扰动 `/Users/dpthinker/workspace/LiteRT-LM` 的 main 工作区。
