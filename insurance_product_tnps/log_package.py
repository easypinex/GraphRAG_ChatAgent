"""
##################################################
## (C) Copyright 2020 TaiwanLife, Inc.
##
## This software is the property of TaiwanLife, Inc.
## You have to accept the terms in the license file before use.
##################################################
"""
import os
import sys
import grp
import socket
import inspect
import logging
import traceback
from functools import wraps
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


log_level_map = {10: logging.DEBUG,
                20: logging.INFO,
                30: logging.WARNING,
                40: logging.ERROR,
                50: logging.CRITICAL}


def settingLogger(logger_name="ACP", 
                    log_level=10, 
                    rotate_when="midnight",
                    code_version="UndefinedCodeVersion", 
                    log_file_path="./Default_Log_Folder/System_log"):
    """
    設定並取得 logger
    
    args:
    logger_name: 本 logger 的名稱, 只要在同一個 processing 都可以透過它取得同一個 logger
    
    log_level: 允許顯示的 log 等級
    
    rotate_when: 多久再次創建一個 log file
    
    code_version: log 內都會帶著這個 code version
    
    log_file_path: log file 預計創建的路徑, 路徑當中所需要的目錄也會一併創建
    """
    log_level_obj = log_level_map[log_level]

    # Logger initial setting
    logger = logging.getLogger(logger_name)
    logger.handlers = []                           # 原本預設都會有一個 StreamHandler, 直接清空

    # Log file setting
    folder_path = os.path.dirname(log_file_path)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    format_pattern = "%(asctime)s [%(levelname)s] [%(p_id)s] [%(func_name)s] [%(file_name)s] %(message)s \n" # builtin keyword: asctime, levelname, message
    formatter = logging.Formatter(fmt=format_pattern, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Setting the logger
    # Create handler to output log on the terminal
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setFormatter(formatter)
    
    # Create handler to store log in the log file
    filehandler = CustomTimedRotatingFileHandler(log_file_path, when=rotate_when, encoding="utf-8")
    filehandler.setFormatter(formatter)

    logger.setLevel(log_level_obj)
    logger.addHandler(streamhandler)                       # Architecture = logger(handler(formatter))
    logger.addHandler(filehandler)
    
    # 將原始的 logging.debug 加上裝飾器, 為了要能夠將 format_pattern 所需要的參數傳進去
    logger.debug = logging_decorator(logger.debug)
    logger.info = logging_decorator(logger.info)
    logger.warning = logging_decorator(logger.warning)
    logger.critical = logging_decorator(logger.critical)
    logger.error = error_logging_decorator(logger.error)
    
    logger.info(f"Code Version:{code_version}, server_ip:{get_host_ip()}")
    
    return logger

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    考慮到體檢報告程式並非批次, 而是啟動服務後常態維持
    如果原本的寫法會讓 log 一直寫在同一個 log file, 然後隨時間越來越肥
    
    因此本 handler 能夠實踐從程式啟動時間開始, 每到達指定的時間間隔(秒, 分, 時, 午夜, 週)會新開一個 log file 並且檔名包含年月日時分秒
    
    但是如果使用原生的 TimedRotatingFileHandler 他會先產生一個預設的檔案, 等到輪轉時間到才會把那個檔案改成有加時間戳記的, 這樣的模式跟理想不符因此修改
    
    原本:
    啟動時產生 default name log file
    輪轉時將這個檔案改名加上時間戳
    再創新的 default name log file
    
    現在:
    啟動時直接產生 default name 並加上時間戳的 log file
    輪轉時就創新的 default name 加上新的時間戳的 log file
    """
    
    def __init__(self, base_filename, when='midnight', interval=1, backupCount=0, encoding=None, delay=False, utc=False, atTime=None, group="docker", chmod=0o775):
        self.base_filename = base_filename
        self.group = group
        self.chmod = chmod
        super().__init__(self._get_filename(), when, interval, backupCount, encoding, delay, utc, atTime)
        self._set_permissions(self.baseFilename)

    def doRollover(self):
        self.baseFilename = self._get_filename()
        super().doRollover()
        self._set_permissions(self.baseFilename)
            
    def _get_filename(self):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"{self.base_filename}_{timestamp}.log"
    
    def _set_permissions(self, filename):
        if self.group:
            gid = grp.getgrnam(self.group).gr_gid
            os.chown(filename, -1, gid)
        if self.chmod:
            os.chmod(filename, self.chmod)
    
def logging_decorator(func):
    @wraps(func)
    def wrapper(msg, *args, **kwargs):
        file_n, func_n, line_num, p_id = get_current_state_by_frame()
        extra = kwargs.pop("extra", {})
        extra.update({
            "file_name": f"{file_n}:{line_num}",
            "func_name": func_n,
            "p_id": f"PID {p_id}",
        })
        func(f"\n{msg}", *args, **kwargs, extra=extra)
    return wrapper

def error_logging_decorator(func):
    @wraps(func)
    def wrapper(msg, error, *args, **kwargs):
        if isinstance(error, Exception):
            msg += f" || {get_traceback()}"
        file_n, func_n, line_num, p_id = get_current_state_by_frame()
        extra = kwargs.pop("extra", {})
        extra.update({
            "file_name": f"{file_n}:{line_num}",
            "func_name": func_n,
            "p_id": f"PID {p_id}",
        })
        func(f"\n{msg}", *args, **kwargs, extra=extra)
    return wrapper

def get_host_ip(): 
    try:
        host_name = socket.gethostname() 
        host_ip = socket.gethostbyname(host_name) 
        return host_ip
    except Exception:
        return "UnknownIP"

def get_current_state_by_frame(layer=2):
    callerframerecord = inspect.stack()[layer]  # layer 0 represents this line
                                                # layer 1 represents file_n, func_n, line_num = get_current_state_by_frame() in decorator
                                                # layer 2 represents the line where logging
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)

    return info.filename, info.function, info.lineno, os.getpid()

def get_traceback():
    return traceback.format_exc()


if __name__ == "__main__":
    """
    使用範例
    """
    import time
    
    # main.py
    CODE_VERSION = "v1.0"
    LOG_FOLDER_PATH = "Demo"
    SYS_LOG_F_NAME = "DemoLog"

    logger = settingLogger(code_version=CODE_VERSION, rotate_when="m", log_file_path=os.path.join(".", LOG_FOLDER_PATH, SYS_LOG_F_NAME))
    logger.info("A")

    try:
        raise Exception("Error Sample")
    except Exception as e:
        logger.error("A2", e)
        
    # 假設在同一個 Process 下的別的地方, 也想沿用同一個 logger
    logger2 = logging.getLogger("ACP")
    logger2.info("B")
    
    class test_log_class:
        def __init__(self) -> None:
            self.logger = logging.getLogger("ACP")
            self.logger.info("C")
    
    obj = test_log_class()
    
    # 測試循環創建 log file
    time_count = 65
    count = 0
    while count < time_count:
        logger.info(f"count: {count}")
        count += 1
        time.sleep(1)
    