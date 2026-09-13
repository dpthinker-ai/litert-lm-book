# 《端侧大模型推理：原理与 LiteRT-LM 实现》

> *On-Device LLM Inference: Principles and Practice with LiteRT-LM*
> **内部打磨版 · 私有仓库 · 非官方**。开源与否未决策前，勿向外部渠道发布任何内容片段。

本书分析大语言模型如何在手机、手表和浏览器等受限设备上运行，
以 Google 的端侧 LLM 运行时 [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) 为主要实现对象。

## 先读这两份

- **`CLAUDE.md`** — 写作规范：每个写作会话逐字加载，含断言四级制、禁词表、除 AI 味清单、图表规范与修订工作流。`AGENTS.md` 只是指向它的指针，供其他代理工具读取。
- **`BOOK_PLAN.md`** — 全书档案：定位、目录（11 章 + 尾声 + 附录 A-E）、当前实测与开放项。

修订记录以 git 提交历史为准，不另写审校文档；初稿期的工作文件冻结在 `archive/`。

## 目录结构

```
cover.md preface.md       封面与前言
parts/                    四个正文部分的导页
chapters/chNN-slug/       每章正文与 SVG 配图
appendix/                 附录 A-E（术语表、代码地图、复现、基准数据集、练习答案）
assets/book.css           SVG 配图的 CSS 变量表（单一事实源）
theme/                    mdBook 主题覆盖
experiments/              实验脚本、采集协议与基准数据（data/）
archive/                  初稿期工作文件归档（已冻结）
.claude/skills/           审校与写作参考 skill（book-review、review-calibration、writing-reference）
.agents/skills/           book-review 的同步副本，供其他代理工具读取
SUMMARY.md book.toml      mdBook 配置
scripts/lint_prose.sh     文风机检（禁词/填充词/内部标签/中英文空格/破折号）
scripts/check_book_consistency.py  SUMMARY、章标题、目录层级与 PDF 单元表一致性
scripts/check_citations.py         页下注完整性与正文、附录间来源去重
scripts/check_code_references.py   源码锚点与代码片段原文检查（对照 v0.17.0 worktree）
scripts/check_book_links.py        成书 HTML 本地链接与锚点检查
scripts/check_pdf_outline.py       PDF 目录与书签检查
scripts/stage_book.sh     白名单 staging：只把正文、附录、图和样式放进书稿源
scripts/build_book.sh     全套检查 + staging + mdBook 构建 + 链接检查
scripts/make_pdf.sh       WebKit 连续排版 + A4 分页 + 页下注 + 目录/书签/页脚
scripts/make_review_page.py        由成书 HTML 生成单章审阅页
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

全书代码引用锁定 `google-ai-edge/LiteRT-LM @ v0.17.0`。核对代码时在**独立 worktree** checkout 该 tag，
不扰动 `/Users/dpthinker/workspace/LiteRT-LM` 的 main 工作区。

本基线来自 `release/v0.17.0`，冻结提交为 `e9fd8c53ff968071774206163027dd84bedfe925`；不跟随 main。
附录 D 与历史实验脚本保留原采集版本，升级后的基础生成检查不构成性能复测。

该 release 的 `WORKSPACE` 固定 LiteRT 提交为 `9fe5be45564c868408e6514c8aabb83e211a0911`；下层算子核对沿用这份依赖，不改用 LiteRT main。
