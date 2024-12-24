from flask_apscheduler import APScheduler
from flask import Blueprint, jsonify

scheduler_module = Blueprint('scheduler', __name__)

# 定義 Scheduler 物件
scheduler = APScheduler()

from logger.logger import get_logger
from scheduler_module.scheduler_tasks import daily_task

from apscheduler.triggers.cron import CronTrigger
logging = get_logger()

# 初始化排程
def init_scheduler(app):
    scheduler.init_app(app)
    scheduler.start()

    # 添加每日執行任務
    scheduler.add_job(
        id='daily_task',  # 任務 ID
        func=daily_task,  # 任務函數路徑
        trigger=CronTrigger.from_crontab('0 21 * * 5')  # 每週五21:00執行
    )

@scheduler_module.route('/start_batch/daily', methods=['POST'])
def start_batch():
    logging.info("start_batch/daily from API.")
    daily_task()
    return jsonify({'message': 'success'})