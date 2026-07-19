#!/usr/bin/env bash
# 文风机检：禁词表、填充词、内部标签、退役叙事词、中英文空格、连续破折号。
# 用法：scripts/lint_prose.sh [文件...]   缺省检查 chapters/**/chapter.md + appendix/*.md + preface.md + cover.md
# 退出码：有命中返回 1，干净返回 0（可挂 pre-commit / CI）。
# 规则依据 AGENTS.md 第三、五节；此脚本只查成书源文件，不查 notes.md（含源码摘录）。
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=()
  while IFS= read -r f; do TARGETS+=("$f"); done < <(find chapters -name 'chapter.md' 2>/dev/null)
  while IFS= read -r f; do TARGETS+=("$f"); done < <(find appendix -name '*.md' 2>/dev/null)
  for f in preface.md cover.md; do [ -f "$f" ] && TARGETS+=("$f"); done
fi
[ "${#TARGETS[@]}" -eq 0 ] && { echo "无 chapter.md 可检查（尚未开始写正文）。"; exit 0; }

python3 - "${TARGETS[@]}" <<'PY'
import sys, re

BANNED = "颠覆 革命性 炸裂 王炸 吊打 秒杀 遥遥领先 天花板 终极 震撼 必看 干货 硬核 赋能 抓手 闭环 完美 极致".split()
FILLER = "值得注意的是 需要指出的是 不难发现 显而易见 可以说 本质上 毫无疑问 众所周知 让我们 综上所述 换句话说".split()
RETIRED = [
    "三堵墙",
    "一个 token 的一生",
    "decode 心跳",
    "decode 的心跳",
    "工单",
    "凿墙",
    "解剖",
    "实剖",
    "标本",
    "鸟瞰",
    "脊柱",
    "真身",
    "还差半步",
    "吐字",
    "占坑",
    "守规矩",
    "万物皆",
    "掐掉",
    "祈祷",
    "一字不改",
]
INTERNAL_LABEL_RE = re.compile(r'【(?:实证|文档|常识|推测)】')

# 中英文之间应留半角空格：CJK 紧邻 ASCII 字母/数字（含反向），排除标点与代码围栏内。
CJK = r'[一-鿿]'
SPACE_RE = re.compile(rf'({CJK}[A-Za-z0-9]|[A-Za-z0-9]{CJK})')
DASH_RE = re.compile(r'——.*——')  # 一段内两个破折号对（粗检，需人工确认是否成对插入语）

hits = 0
def report(f, ln, kind, detail):
    global hits; hits += 1
    print(f"{f}:{ln}: [{kind}] {detail}")

for path in sys.argv[1:]:
    try:
        lines = open(path, encoding='utf-8').read().splitlines()
    except FileNotFoundError:
        print(f"跳过（不存在）：{path}"); continue
    in_code = False
    in_comment = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith('```'):
            in_code = not in_code; continue
        if in_code:
            continue
        # HTML 注释不会出现在成书中；审校历史与待办项另由 review.md 承载。
        visible = line
        if in_comment:
            if '-->' in visible:
                visible = visible.split('-->', 1)[1]
                in_comment = False
            else:
                continue
        while '<!--' in visible:
            before, after = visible.split('<!--', 1)
            if '-->' in after:
                visible = before + after.split('-->', 1)[1]
            else:
                visible = before
                in_comment = True
                break
        for w in BANNED:
            if w in visible: report(path, i, "禁词", w)
        for w in FILLER:
            if w in visible: report(path, i, "填充词", w)
        for w in RETIRED:
            if w in visible: report(path, i, "退役叙事词", w)
        label = INTERNAL_LABEL_RE.search(visible)
        if label:
            report(path, i, "内部标签", label.group(0))
        # 中英文空格：跳过行内代码 `...` 与链接/路径
        probe = re.sub(r'`[^`]*`', '', visible)
        probe = re.sub(r'\[[^\]]*\]\([^)]*\)', '', probe)
        if SPACE_RE.search(probe):
            m = SPACE_RE.search(probe)
            report(path, i, "中英文空格", m.group(0))
        if DASH_RE.search(visible):
            report(path, i, "破折号", "一段内出现两处破折号（成对插入语算一处，请人工确认）")

print("-" * 40)
if hits:
    print(f"共 {hits} 处命中，请逐条处理（空格/破折号需人工确认，其余必改）。")
    sys.exit(1)
print("文风机检通过：0 命中。")
PY
