#!/data/data/com.termux/files/usr/bin/bash

echo "======================================"
echo "   凄凉Tool 环境准备与启动脚本"
echo "======================================"

# 定义颜色输出（和Python脚本保持一致）
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
NC="\033[0m"

# 更新 Termux 包管理器
echo -e "${BLUE}更新 Termux 包...${NC}"
pkg update -y && pkg upgrade -y

# 检查并安装 Python
if ! command -v python &> /dev/null; then
    echo -e "${BLUE}安装 Python...${NC}"
    pkg install python -y
else
    echo -e "${GREEN}✓ Python 已安装${NC}"
fi

# 检查并安装 pip
if ! command -v pip &> /dev/null; then
    echo -e "${BLUE}安装 pip...${NC}"
    pkg install python-pip -y
else
    echo -e "${GREEN}✓ pip 已安装${NC}"
fi

# 升级 pip
echo -e "${BLUE}升级 pip...${NC}"
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装必要的系统包
echo -e "${BLUE}安装其他依赖...${NC}"
pkg install clang libffi openssl -y

# 检查并安装 Python 依赖
echo -e "${BLUE}检查 Python 依赖库...${NC}"

REQUIRED_PACKAGES=(
    "gmalg"
    "pycryptodome"
    "zstandard"
    "packaging"
)

# 清华源地址（提速安装）
PYPI_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --quiet"

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" &>/dev/null; then
        echo -e "${BLUE}正在安装必要依赖...${NC}"
        pip install "$package" $PYPI_MIRROR
    else
        echo -e "${GREEN}✓ $package 已安装${NC}"
    fi
done

echo "======================================"
echo -e "${GREEN}依赖检查完成，准备启动 凄凉Tool...${NC}"
echo "======================================"

# 检查 Shrem.py 是否存在
if [ ! -f "凄凉Tool.py" ]; then
    echo -e "${RED}错误：在当前目录找不到 凄凉Tool.py${NC}"
    echo -e "${YELLOW}请将 凄凉Tool.py 放在当前目录后再运行此脚本${NC}"
    exit 1
fi

# 运行主程序
echo -e "${BLUE}启动 凄凉Tool...${NC}"
chmod +x 凄凉Tool.py
./凄凉Tool.py
