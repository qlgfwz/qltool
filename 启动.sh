#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail # 快速错误终止，减少冗余执行

# 颜色定义
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
NC="\033[0m"

# 配置（可自行开关）
SKIP_PKG_UPDATE=1  # 1=关闭Termux包更新（提速核心），0=开启
CACHE_DEP_CHECK=1  # 1=缓存依赖检测结果，0=每次检测
DEP_CACHE_FILE=~/.qltool_dep_cache # 依赖缓存文件路径
SCRIPT_NAME="凄凉Tool.py"          # 主脚本名
#!/bin/bash
# 获取当前sh脚本的绝对路径
#!/bin/bash
# 获取当前sh脚本绝对路径（匹配Python Path(__file__).resolve()）
SCRIPT_PATH=$(readlink -f "$0")
# 当前脚本所在目录（仅检查/删除此目录下的qltool）
CURRENT_DIR=$(dirname "$SCRIPT_PATH")
TARGET="qltool"
TARGET_PATH="${CURRENT_DIR}/${TARGET}"

# 检查并删除#!/bin/bash
# 颜色定义（基础色+高亮色，和Python代码统一）
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
PURPLE="\033[35m"  # 网速数值高亮色
NC="\033[0m"        # 重置颜色

# 网络连通性检测（ping百度DNS，超时3秒，2次包，屏蔽冗余输出）
check_network() {
    echo -e "${BLUE}🔍 正在检测网络连通性...${NC}"
    if ping -c 2 -W 3 180.76.76.76 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 网络连通，开始测试网速...${NC}\n"
        return 0
    else
        echo -e "${RED}❌ 未连接网络，无法测试网速${NC}"
        return 1
    fi
}

main() {
    check_network
}

main

if [ -e "$TARGET_PATH" ]; then
    rm -rf "$TARGET_PATH"
    echo "✅ 已删除当前脚本目录[${CURRENT_DIR}]下的${TARGET}"
else
    echo "❌ 当前脚本目录[${CURRENT_DIR}]下无${TARGET}，无需删除"
fi

echo -e "${BLUE}更新 Termux 包...${NC}"
pkg update -y && pkg upgrade -y

# 快速打印标题
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   凄凉Tool 极速启动脚本${NC}"
echo -e "${BLUE}======================================${NC}\n"

# -------------------------- 1. 极简系统依赖检查（仅首次/强制检测）--------------------------
if [ $CACHE_DEP_CHECK -eq 1 ] && [ ! -f "$DEP_CACHE_FILE" ]; then
    echo -e "${YELLOW}🔍 首次运行，检测基础环境...${NC}"
    echo -e "${BLUE}更新 Termux 包...${NC}"
pkg update -y && pkg upgrade -y

    if ! command -v python3 &> /dev/null; then
        echo -e "${BLUE}📦 安装Python3...${NC}"
        pkg install python -y --quiet
    fi
    # 检查pip（无则安装，不升级）
    if ! command -v pip &> /dev/null; then
        echo -e "${BLUE}📦 安装pip...${NC}"
        pkg install python-pip -y --quiet
    fi
    # 安装必要系统包（静默安装）
    echo -e "${BLUE}📦 安装基础编译依赖...${NC}"
    pkg install clang libffi openssl -y --quiet &> /dev/null
    # 创建缓存文件，标记环境检测完成
    touch "$DEP_CACHE_FILE"
    echo -e "${GREEN}✅ 基础环境检测完成，后续启动将跳过此步骤\n${NC}"
elif [ $CACHE_DEP_CHECK -eq 0 ]; then
    echo -e "${YELLOW}🔍 强制检测Python3...${NC}"
    command -v python3 &> /dev/null || (pkg install python -y --quiet)
fi

# -------------------------- 2. 极简Python依赖检查（仅首次/强制检测）--------------------------
REQUIRED_PACKAGES=("gmalg" "pycryptodome" "zstandard" "packaging" "requests")
PYPI_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --quiet --no-cache-dir"

if [ $CACHE_DEP_CHECK -eq 1 ] && [ ! -f "$DEP_CACHE_FILE" ]; then
    echo -e "${YELLOW}🔍 检测Python依赖库...${NC}"
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! pip show "$package" &>/dev/null; then
            pip install "$package" $PYPI_MIRROR &> /dev/null
            echo -e "${GREEN}✓ 安装${package}成功${NC}"
        fi
    done
    echo -e "${GREEN}✅ Python依赖检测完成\n${NC}"
elif [ $CACHE_DEP_CHECK -eq 0 ]; then
    echo -e "${YELLOW}🔍 强制检测Python依赖...${NC}"
    for package in "${REQUIRED_PACKAGES[@]}"; do
        pip show "$package" &>/dev/null || pip install "$package" $PYPI_MIRROR &> /dev/null
    done
fi

# -------------------------- 3. 主脚本检查 + 极速启动 --------------------------
# 检查主脚本是否存在
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${RED}❌ 错误：当前目录未找到${SCRIPT_NAME}${NC}"
    exit 1
fi

# 赋予执行权限（仅首次）
[ ! -x "$SCRIPT_NAME" ] && chmod +x "$SCRIPT_NAME"

# 优先./运行，失败兜底python3（极速执行，无冗余输出）
echo -e "${BLUE}🚀 启动凄凉Tool...${NC}\n"

if ./凄凉Tool.py; then
    echo -e "${GREEN}✓ 凄凉Tool 运行成功${NC}"
else
    echo -e "${YELLOW}⚠️  直接运行失败，尝试用 Python3 启动...${NC}"
    if python3 凄凉Tool.py; then
        echo -e "${GREEN}✓ Python3 启动 凄凉Tool 成功${NC}"
    else
        echo -e "${RED}✗ 所有启动方式均失败，请检查脚本或依赖${NC}"
        exit 1
    fi
fi



