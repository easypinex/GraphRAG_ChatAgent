#!/bin/bash

# 獲取當前 Shell 腳本的絕對路徑
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd ${script_dir}

# 獲取當前目錄下的所有文件夾
directories=$(find . -maxdepth 1 -type d)

# 依序進入每個文件夾並執行 docker-compose up -d
for dir in $directories; do
    # 排除當前目錄
    if [ "$dir" != "." ]; then
        echo "Entering directory: $dir"
        cd "$dir"
        if [ -f "docker-compose.yml" ]; then
            echo "Running docker-compose up -d in $dir"
            docker-compose down
        else
            echo "No docker-compose.yml found in $dir, skipping..."
        fi
        cd -  # 返回上一個目錄
    fi
done