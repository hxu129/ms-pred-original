#!/bin/bash
# 统一环境激活脚本 (Unified Environment Activation Script)

# 初始化 conda/mamba（如果尚未完成）
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# 激活环境
mamba activate unified-ms-env

echo "✓ 统一 MS 环境已激活！(Unified MS environment activated!)"
echo ""
echo "环境信息 (Environment details):"
echo "  - Python: $(python --version 2>&1)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  - CUDA 可用 (CUDA available): $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo ""
echo "可用项目 (Available projects):"
echo "  - DiffMS: /root/ms/DiffMS"
echo "  - ms-pred: /root/ms/ms-pred"
echo ""
echo "文档 (Documentation):"
echo "  - 环境重建指南: /root/ms/环境重建指南.md"
echo "  - 详细文档: /root/ms/UNIFIED_ENVIRONMENT_SETUP_SUMMARY.md"
echo ""
echo "退出环境 (To deactivate): mamba deactivate"

