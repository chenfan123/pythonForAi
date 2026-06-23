#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "未找到 .venv，请先运行: make install"
  exit 1
fi

# Apple Silicon：虚拟环境为 arm64，检测与执行均走原生架构
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  PYTHON=(arch -arm64 .venv/bin/python)
else
  PYTHON=(.venv/bin/python)
fi

TERM_ARCH="$(uname -m)"
VENV_ARCH="$("${PYTHON[@]}" -c 'import platform; print(platform.machine())')"

if [ "$TERM_ARCH" != "$VENV_ARCH" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]; then
  echo "错误：终端架构 ($TERM_ARCH) 与虚拟环境 ($VENV_ARCH) 不一致。"
  echo "请关闭「使用 Rosetta 打开」后重建: rm -rf .venv && make install"
  exit 1
fi

exec "${PYTHON[@]}" "$@"
