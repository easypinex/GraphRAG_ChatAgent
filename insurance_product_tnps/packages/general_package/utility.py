
import json
import psutil


def check_memory(threshold_GB=72):
    """
    檢查系統記憶體大小
    返回：
        bool: 如果記憶體 >= threshold_GB GB 返回 True，否則返回 False
    """
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)  # 轉換為 GB
    return memory_gb >= threshold_GB

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=lambda o: o.to_dict())

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
