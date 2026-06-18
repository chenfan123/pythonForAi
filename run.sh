#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "未找到 .venv，请先运行: make install"
  exit 1
fi

TERM_ARCH="$(uname -m)"
VENV_ARCH="$(.venv/bin/python -c 'import platform; print(platform.machine())')"

if [ "$TERM_ARCH" != "$VENV_ARCH" ]; then
  echo "错误：终端架构 ($TERM_ARCH) 与虚拟环境 ($VENV_ARCH) 不一致。"
  echo "请任选一种方式修复："
  echo "  1) 关闭终端 Rosetta，使用原生 arm64 终端后运行: rm -rf .venv && make install"
  echo "  2) 在 Rosetta 终端中重建: rm -rf .venv && arch -x86_64 python3.14 -m venv .venv && arch -x86_64 make install"
  exit 1
fi

exec .venv/bin/python "$@"
