#!/usr/bin/env python3
"""在不同代理环境下探测 litert-lm CLI 的 Hugging Face 下载路径。

用法：<venv>/bin/python experiments/hf_download_proxy_probe.py --port <代理端口> [--repo <仓库>]
对四种代理环境各启动一个子进程（同一解释器），调用 CLI 自带的下载模块列出仓库文件并下载
README.md，记录异常类型、消息与耗时；不下载模型文件。兼容 v0.13.1（common.download_from_huggingface，
经 huggingface_hub/httpx）与 v0.14.0 起的 litert_lm_cli.huggingface_download（标准库 urllib）。
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import contextlib, importlib, io, json, os, socket, sys, time
socket.setdefaulttimeout(20)
repo = sys.argv[1]
out = {"proxy_env": {k: v for k, v in os.environ.items() if k.lower().endswith("_proxy")}}
import urllib.request
out["urllib_getproxies"] = urllib.request.getproxies()
try:
    _t0 = time.monotonic()
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with _opener.open("https://huggingface.co/api/models/" + repo) as _r:
        out["direct_no_proxy"] = {"ok": True, "status": _r.status, "seconds": round(time.monotonic() - _t0, 2)}
except BaseException as e:
    out["direct_no_proxy"] = {"ok": False, "error": type(e).__name__, "message": str(e)[:200], "seconds": round(time.monotonic() - _t0, 2)}
for mod in ("httpx", "socksio", "huggingface_hub"):
    try:
        m = importlib.import_module(mod)
        out["has_" + mod] = getattr(m, "__version__", True)
    except ImportError:
        out["has_" + mod] = False

def timed(name, fn):
    t0 = time.monotonic()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            r = fn()
        rec = {"ok": r is not None, "result": str(r)[:160]}
    except BaseException as e:  # 记录任何异常类型
        rec = {"ok": False, "error": type(e).__name__, "message": str(e)[:300]}
    rec["seconds"] = round(time.monotonic() - t0, 2)
    printed = buf.getvalue().strip()
    if printed:
        rec["printed"] = printed[-300:]
    out[name] = rec

try:
    from litert_lm_cli import huggingface_download as hd
    out["module"] = "litert_lm_cli.huggingface_download (urllib)"
    timed("list_files", lambda: hd.list_litertlm_files(repo_id=repo, token=None))
    timed("download_readme", lambda: hd.download_from_huggingface(repo_id=repo, filename="README.md", token=None))
except ImportError:
    from litert_lm_cli import common
    out["module"] = "litert_lm_cli.common (huggingface_hub)"
    timed("download_readme", lambda: common.download_from_huggingface(repo, "README.md", None))
print(json.dumps(out, ensure_ascii=False))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True, help="本机代理端口（HTTP 与 SOCKS 共用或分别指定）")
    ap.add_argument("--socks-port", type=int, default=None, help="SOCKS 端口，缺省与 --port 相同")
    ap.add_argument("--repo", default="litert-community/gemma-4-E4B-it-litert-lm")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--cli-cache", default="~/.litert-lm/cache/huggingface",
                    help="v0.14.0+ CLI 的下载缓存根目录；每种环境前删除其中的 README.md，保证下载步骤真正走网络")
    a = ap.parse_args()
    socks_port = a.socks_port or a.port
    base = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    http = f"http://127.0.0.1:{a.port}"
    socks = f"socks5://127.0.0.1:{socks_port}"
    cases = {
        "A-http-proxy-env": {**base, "HTTP_PROXY": http, "HTTPS_PROXY": http, "http_proxy": http, "https_proxy": http},
        "B-no-proxy-env": base,
        "C-socks-only-ALL_PROXY": {**base, "ALL_PROXY": socks, "all_proxy": socks},
        "D-https_proxy-socks-url": {**base, "HTTPS_PROXY": socks, "https_proxy": socks},
    }
    results = {"python": sys.executable, "http_proxy": http, "socks_proxy": socks, "repo": a.repo, "cases": {}}
    cached_readme = os.path.join(os.path.expanduser(a.cli_cache), a.repo, "README.md")
    for name, env in cases.items():
        if os.path.exists(cached_readme):
            os.remove(cached_readme)
        try:
            p = subprocess.run([sys.executable, "-c", CHILD, a.repo], env=env,
                               capture_output=True, text=True, timeout=a.timeout)
            try:
                rec = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:
                rec = {"parse_error": True, "returncode": p.returncode,
                       "stdout": p.stdout[-400:], "stderr": p.stderr[-600:]}
        except subprocess.TimeoutExpired:
            rec = {"timeout_seconds": a.timeout}
        results["cases"][name] = rec
        brief = {k: rec.get(k) for k in ("module", "direct_no_proxy", "list_files", "download_readme", "timeout_seconds", "parse_error") if k in rec}
        print(f"[{name}] {json.dumps(brief, ensure_ascii=False)}")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
