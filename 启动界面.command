#!/bin/bash
cd "$(dirname "$0")"

# 优先使用配置好的 Conda 环境，否则回退至系统 Python3
if [ -f "/opt/miniconda3/envs/py_course/bin/python" ]; then
    /opt/miniconda3/envs/py_course/bin/python gui_app.py
elif which python3 >/dev/null 2>&1; then
    python3 gui_app.py
else
    python gui_app.py
fi
