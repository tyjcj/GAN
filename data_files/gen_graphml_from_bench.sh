#!/bin/bash

# 批量转换脚本：将openabcd文件夹下所有.bench文件转换为.graphml格式
# 依赖：提供的andAIG2Graphml.py脚本

# 配置参数
PYTHON_SCRIPT="andAIG2Graphml.py"  # Python转换脚本路径
ROOT_DIR="openabcd/bench"                # 根目录
GML_OUTPUT_DIR="openabcd/graphml"  # 输出目录

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误：Python脚本 $PYTHON_SCRIPT 不存在！"
    exit 1
fi

# 检查根目录是否存在
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误：根目录 $ROOT_DIR 不存在！"
    exit 1
fi

# 创建输出目录
mkdir -p "$GML_OUTPUT_DIR"
echo "输出目录：$GML_OUTPUT_DIR"

# 统计待处理文件数量
BENCH_COUNT=$(find "$ROOT_DIR" -type f -name "*.bench" | wc -l)
echo "发现 $BENCH_COUNT 个bench文件，开始转换..."

# 批量转换
find "$ROOT_DIR" -type f -name "*.bench" | while read -r bench_file; do
    # 提取文件名（不含路径）
    filename=$(basename "$bench_file")
    echo -e "\n处理文件：$filename"

    # 执行转换
    python3 "$PYTHON_SCRIPT" \
        --bench "$bench_file" \
        --gml "$GML_OUTPUT_DIR"

    # 检查转换结果
    if [ $? -eq 0 ]; then
        echo "✅ 转换成功：$filename -> ${filename%}.graphml"
    else
        echo "❌ 转换失败：$filename"
    fi
done

echo -e "\n批量转换完成！"
echo "转换结果保存在：$GML_OUTPUT_DIR"
