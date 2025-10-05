#!/bin/bash

# 批量转换脚本：将graphml文件夹下所有.graphml文件转换为.pt格式
# 依赖：提供的generate_pt.py脚本
# 命名规则：xxx.bench.graphml -> xxx.pt

# 配置参数
PYTHON_SCRIPT="data_files/generate_pt.py"       # Python转换脚本路径
GRAPHML_DIR="data_files/datasets/ISCAS85/graphml"       # graphml文件所在目录
OUTPUT_DIR="data_files/datasets/ISCAS85/graph"                # 输出pt文件的目录

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误：Python脚本 $PYTHON_SCRIPT 不存在！"
    exit 1
fi

# 检查graphml目录是否存在
if [ ! -d "$GRAPHML_DIR" ]; then
    echo "错误：graphml目录 $GRAPHML_DIR 不存在！"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"
echo "输出目录：$OUTPUT_DIR"

# 统计待处理文件数量
GRAPHML_COUNT=$(find "$GRAPHML_DIR" -maxdepth 1 -type f -name "*.bench.graphml" | wc -l)
echo "发现 $GRAPHML_COUNT 个符合格式的graphml文件，开始转换..."

# 批量转换
find "$GRAPHML_DIR" -maxdepth 1 -type f -name "*.bench.graphml" | while read -r graphml_file; do
    # 提取文件名（不含路径）
    filename=$(basename "$graphml_file")

    # 生成输出文件名：xxx.bench.graphml -> xxx.pt
    output_filename=$(echo "$filename" | sed 's/\.bench\.graphml$/.pt/')
    output_path="$OUTPUT_DIR/$output_filename"

    echo -e "\n处理文件：$filename"
    echo "输出文件：$output_filename"

    # 执行转换
    python3 "$PYTHON_SCRIPT" \
        --graphml "$graphml_file" \
        --out "$output_path"

    # 检查转换结果
    if [ $? -eq 0 ] && [ -f "$output_path" ]; then
        echo "✅ 转换成功"
    else
        echo "❌ 转换失败"
    fi
done

echo -e "\n批量转换完成！"
echo "转换结果保存在：$OUTPUT_DIR"
