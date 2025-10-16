# 快速参考卡片 (Quick Reference Card)

## 🚀 环境激活 (Environment Activation)

```bash
# 方法 1：使用激活脚本
source /root/ms/activate_unified_env.sh

# 方法 2：直接激活
source ~/.bashrc
mamba activate unified-ms-env
```

## 📦 环境信息 (Environment Info)

| 项目 | 版本 | 说明 |
|------|------|------|
| **环境名称** | unified-ms-env | 统一开发环境 |
| **Python** | 3.9.23 | 兼容两个项目 |
| **PyTorch** | 2.3.1+cu118 | 支持 CUDA 11.8 |
| **PyTorch Lightning** | 2.0.4 | 从 1.6 升级 |
| **RDKit** | 2024.09.4 | 从 2021.03 升级 |

## 📂 重要文件 (Important Files)

| 文件 | 描述 |
|------|------|
| `/root/ms/环境重建指南.md` | **完整重建指南** |
| `/root/ms/UNIFIED_ENVIRONMENT_SETUP_SUMMARY.md` | 详细设置摘要 |
| `/root/ms/unified_environment.yml` | Conda 环境定义 |
| `/root/ms/unified_requirements.txt` | Python 依赖列表 |
| `/root/ms/activate_unified_env.sh` | 环境激活脚本 |

## 🔧 运行项目 (Running Projects)

### DiffMS

```bash
cd /root/ms/DiffMS
python src/fp2mol_main.py      # 指纹到分子
python src/spec2mol_main.py    # 光谱到分子
```

### ms-pred

```bash
cd /root/ms/ms-pred

# ICEBERG 模型
bash run_scripts/iceberg/01_run_dag_gen_train.sh

# MARASON 模型
bash run_scripts/marason/01_run_marason_gen_train.sh

# SCARF 模型
bash run_scripts/scarf_model/01_run_scarf_gen_train.sh
```

## 🧪 验证命令 (Verification Commands)

### 快速验证
```bash
# 一键验证所有核心组件
python -c "
import torch, pytorch_lightning, torch_geometric, dgl, rdkit
print('✓ 所有核心包导入成功')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### 验证项目导入
```bash
# DiffMS
python -c "from src import utils; print('✓ DiffMS OK')"

# ms-pred
python -c "import ms_pred; print('✓ ms-pred OK')"
```

### GPU 测试
```bash
python -c "
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.mm(x, x)
print(f'✓ GPU 工作正常: {torch.cuda.get_device_name(0)}')
"
```

## 🔄 环境重建 (Environment Reconstruction)

### 在新机器上快速重建

```bash
# 1. 复制文件到新机器
rsync -av /root/ms/ new-machine:/root/ms/

# 2. 在新机器上安装 Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source ~/.bashrc

# 3. 按照 /root/ms/环境重建指南.md 中的步骤操作
```

**详细步骤请参考：** `/root/ms/环境重建指南.md`

## 💾 导出环境 (Export Environment)

### 导出到文件（在迁移前执行）

```bash
# 激活环境
mamba activate unified-ms-env

# 导出完整环境
mamba env export > unified-ms-env-backup-$(date +%Y%m%d).yml

# 导出跨平台配置
mamba env export --from-history > unified-ms-env-simple.yml

# 导出 pip 包
pip list --format=freeze > pip-backup-$(date +%Y%m%d).txt
```

## ⚠️ 已知问题 (Known Issues)

### 1. DiffMS 导入问题
```python
# ❌ 错误
import diffms

# ✅ 正确
from src import utils
```

### 2. CUDA 内存不足
**解决方案：** 在训练脚本中减小 batch_size

### 3. PyTorch Lightning 警告
**原因：** API 已更新到 2.0.4  
**状态：** 已修复（13 个文件已更新）

## 📊 代码修改记录 (Code Modifications)

### ms-pred 项目修改

1. **Cython 修复**
   - 文件：`src/ms_pred/massformer_pred/massformer_code/algos2.pyx`
   - 修改：`long` → `int64_t`

2. **PyTorch Lightning API 更新**（13 个文件）
   - 修改：`.best_model_score.item()` → `.best_model_score`
   - 详见：`/root/ms/环境重建指南.md` 第五步

## 🎯 常用命令 (Common Commands)

```bash
# 激活环境
source /root/ms/activate_unified_env.sh

# 检查 GPU
nvidia-smi

# 查看环境包列表
mamba list

# 更新单个包
pip install --upgrade <package-name>

# 退出环境
mamba deactivate

# 删除环境（谨慎！）
mamba env remove -n unified-ms-env
```

## 📞 获取帮助 (Getting Help)

- **环境重建：** 参见 `/root/ms/环境重建指南.md`
- **详细文档：** 参见 `/root/ms/UNIFIED_ENVIRONMENT_SETUP_SUMMARY.md`
- **DiffMS 文档：** 参见 `/root/ms/DiffMS/README.md`
- **ms-pred 文档：** 参见 `/root/ms/ms-pred/README.md`

---

**最后更新：** 2025-10-16  
**环境位置：** `/root/miniforge3/envs/unified-ms-env`

