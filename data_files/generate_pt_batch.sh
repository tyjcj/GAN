#!/bin/bash

# 定义输入bench文件夹和输出pt文件夹
BENCH_DIR="datasets/openabcd/graphml"
OUTPUT_DIR="datasets/openabcd/graph"

# 创建输出文件夹（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 检查输入文件夹是否存在
if [ ! -d "$BENCH_DIR" ]; then
    echo "错误：输入文件夹 $BENCH_DIR 不存在！"
    exit 1
fi

# 检查generate_pt.py是否存在
if [ ! -f "generate_pt.py" ]; then
    echo "错误：generate_pt.py 文件不存在！"
    exit 1
fi

# 遍历bench文件夹下所有.bench文件
count=0
for bench_file in "$BENCH_DIR"/*.bench; do
    # 检查是否有匹配的文件
    if [ ! -f "$bench_file" ]; then
        echo "在 $BENCH_DIR 目录下未找到.bench文件"
        exit 0
    fi

    # 提取文件名（不含路径和扩展名）
    filename=$(basename -- "$bench_file")
    filename_noext="${filename%.bench}"

    # 定义输出pt文件路径
    output_pt="$OUTPUT_DIR/$filename_noext.pt"

    # 执行转换命令
    echo "正在处理: $filename"
    python generate_pt.py --bench "$bench_file" --out "$output_pt"

    # 检查命令是否执行成功
    if [ $? -eq 0 ]; then
        ((count++))
        echo "已生成: $output_pt"
    else
        echo "处理 $filename 时出错！"
    fi
done

echo "批量处理完成，共成功生成 $count 个.pt文件，保存在 $OUTPUT_DIR 目录下"

