#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import glob
from datetime import datetime

def tail_log(log_path, n=10, follow=True, refresh_rate=1.0):
    """
    類似於 `tail -f` 命令，用於實時顯示日誌文件的最新內容
    
    參數:
        log_path: 日誌文件路徑（可以是模式匹配）
        n: 顯示最後n行
        follow: 是否持續監控文件更新
        refresh_rate: 刷新率（秒）
    """
    def get_latest_log(pattern):
        logs = glob.glob(pattern)
        if not logs:
            return None
        return max(logs, key=os.path.getctime)
    
    # 如果是模式，找到最新的日誌文件
    if '*' in log_path:
        log_file = get_latest_log(log_path)
        if not log_file:
            print(f"找不到匹配 '{log_path}' 的日誌文件")
            return
    else:
        log_file = log_path
        if not os.path.exists(log_file):
            print(f"日誌文件 '{log_file}' 不存在")
            return
    
    print(f"監控日誌文件: {log_file}")
    
    # 初始顯示最後n行
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                print(''.join(lines[-n:]), end='')
    except Exception as e:
        print(f"讀取日誌文件時出錯: {e}")
        return
    
    # 如果不需要持續監控，直接返回
    if not follow:
        return
    
    # 獲取文件大小
    file_size = os.path.getsize(log_file)
    
    print("\n按 Ctrl+C 停止監控...\n")
    
    try:
        while True:
            # 檢查文件是否更新
            current_size = os.path.getsize(log_file)
            if current_size > file_size:
                with open(log_file, 'r', encoding='utf-8') as f:
                    f.seek(file_size)
                    new_content = f.read()
                    print(new_content, end='')
                file_size = current_size
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        print("\n停止監控")
    except Exception as e:
        print(f"監控過程中出錯: {e}")

def main():
    parser = argparse.ArgumentParser(description='實時監控訓練日誌')
    parser.add_argument('log_path', nargs='?', default='training_*.log',
                        help='日誌文件路徑 (默認: training_*.log)')
    parser.add_argument('-n', '--lines', type=int, default=10,
                        help='初始顯示的行數 (默認: 10)')
    parser.add_argument('-f', '--follow', action='store_true', default=True,
                        help='持續監控文件更新 (默認: True)')
    parser.add_argument('-r', '--refresh-rate', type=float, default=0.5,
                        help='刷新率，秒 (默認: 0.5)')
    
    args = parser.parse_args()
    
    tail_log(args.log_path, args.lines, args.follow, args.refresh_rate)

if __name__ == '__main__':
    main()