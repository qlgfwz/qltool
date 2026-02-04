#!/data/data/com.termux/files/usr/bin/bash

echo "======================================"
echo "   凄凉Tool 环境准备与启动脚本"
echo "======================================"

# 更新 Termux 包管理器
echo "更新 Termux 包..."
pkg update -y && pkg upgrade -y

# 检查并安装 Python
if ! command -v python &> /dev/null; then
    echo "安装 Python..."
    pkg install python -y
else
    echo "✓ Python 已安装"
fi

# 检查并安装 pip
if ! command -v pip &> /dev/null; then
    echo "安装 pip..."
    pkg install python-pip -y
else
    echo "✓ pip 已安装"
fi

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装必要的系统包
echo "安装其他依赖..."
pkg install clang libffi openssl -y

# 检查并安装 Python 依赖
echo "检查 Python 依赖库..."

REQUIRED_PACKAGES=(
    "gmalg"
    "pycryptodome"
    "zstandard"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" &>/dev/null; then
        echo "安装 $package..."
        pip install "$package"
    else
        echo "✓ $package 已安装"
    fi
done

echo "======================================"
echo "依赖检查完成，准备启动 凄凉Tool..."
echo "======================================"

# 检查 Shrem.py 是否存在
if [ ! -f "凄凉Tool.py" ]; then
    echo "错误：在当前目录找不到 凄凉Tool.py"
    echo "请将 凄凉Tool.py 放在当前目录后再运行此脚本"
    exit 1
fi

# 运行主程序
echo "启动 凄凉Tool..."
chmod +x 凄凉Tool.py
./凄凉Tool.py