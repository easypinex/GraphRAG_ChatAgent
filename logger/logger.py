import logging
from logging.handlers import TimedRotatingFileHandler
import os

# Logger 初始化
def get_logger(name: str = "chat_agent") -> logging.Logger:
    """
    初始化 Logger 並返回共享的 logger 實例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # 防止多次添加 handler
        logger.setLevel(logging.DEBUG)
        # StreamHandler
        stream_handler = logging.StreamHandler()
        FORMAT = "%(asctime)s - [%(filename)s:%(lineno)s  %(funcName)20s()] %(message)s"
        formatter = logging.Formatter(FORMAT)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # FileHandler
        log_dir = os.getenv("LOG_DIR") or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",       # 每天午夜分割
            interval=1,            # 每 1 天輪轉一次
            backupCount=30          # 保留最近 30 天的日志文件
        )
        handler.suffix = "%Y-%m-%d"  # 文件名包含日期
        formatter = logging.Formatter(FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    # 簡易測試
    logger = get_logger()
    def test_log_func():
        logger.info("This is an info message")
        logger.debug("This is a debug message")
    test_log_func()