
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
