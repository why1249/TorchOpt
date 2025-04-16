# 安装指南

## 环境要求

在安装 TorchOptLib 之前，请确保您具备以下依赖项：

- Python >= 3.9
- PyTorch >= 2.0
- NumPy
- tqdm

## 安装方法

### 使用 pip 安装（即将推出）

```bash
pip install torchoptlib
```

### 从源代码安装

克隆仓库并以开发模式安装：

```bash
git clone https://github.com/why1249/TorchOptLib.git
cd TorchOptLib
pip install -e .
```

## 验证安装

要验证 TorchOptLib 是否已正确安装，您可以运行以下 Python 代码：

```python
import torchoptlib
print(torchoptlib.__version__)
```

## 故障排除

如果您在安装过程中遇到任何问题：

1. 确保您使用的是正确的 Python 版本（3.9+）
2. 验证 PyTorch 是否正确安装
3. 检查安装过程中是否有任何错误消息

如需其他帮助，请在 [GitHub 仓库](https://github.com/why1249/TorchOptLib) 上提出问题。
