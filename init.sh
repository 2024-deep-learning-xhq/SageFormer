#!/bin/bash

# 设置目录路径
DATA_DIR="logs/LongForecasting"

# 检查目录是否存在，如果不存在则创建
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 设置目录路径
DATA_DIR="dataset/weather"

# 检查目录是否存在，如果不存在则创建
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 示例数据下载URL（这里使用一个示例链接，需要替换为实际的数据源）
DATA_URL="https://cloud.tsinghua.edu.cn/f/d343a44026c24a688add/?dl=1"
OUTPUT_FILE="$DATA_DIR/5min.csv"

# 检查wget是否已安装
if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Please install wget first."
    exit 1
fi

# 下载数据文件
echo "Downloading data to $OUTPUT_FILE"
if wget -q --show-progress "$DATA_URL" -O "$OUTPUT_FILE"; then
    echo "Download completed successfully"
    echo "Data file saved to: $OUTPUT_FILE"
else
    echo "Error: Failed to download the data file"
    exit 1
fi