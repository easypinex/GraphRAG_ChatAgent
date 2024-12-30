
from flask import Blueprint, request, jsonify
from datetime import datetime

from dataflow_module.rabbitmq_task import QueueTaskDict
from minio_module.minio_service import minio_service

from dataflow_module.rabbitmq_sender import publish_queue_message

neo4j_module = Blueprint('ne4j_module', __name__)

@neo4j_module.route('/restore', methods=['POST'])
def restore_neo4j():
    try:
        # Extract 'date' from JSON payload
        data = request.get_json()
        date_str = data.get('date')  # Expecting {"date": "yyyymmdd"}
        
        # Validate and parse the date
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        
        exist = minio_service.check_neo4j_backup_file_exist(date_str)
        if not exist:
            return jsonify({
                "status": "error",
                "message": f"DB backup file not found for date: {date_obj.strftime('%Y-%m-%d')}"
            }), 400
        publish_queue_message(QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.RESTORE_NEO4J, msg=date_str))
        
        # Example response: Return formatted date
        return jsonify({
            "status": "success",
            "message": f"Received date: {date_obj.strftime('%Y-%m-%d')}"
        }), 200
    except (ValueError, TypeError, KeyError) as e:
        return jsonify({
            "status": "error",
            "message": "Invalid date format. Use 'yyyymmdd'."
        }), 400
        
@neo4j_module.route('/backup', methods=['POST'])
def backup_neo4j():
    try:
        # Publish a backup task to the queue
        publish_queue_message(QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.BACKUP_NEO4J))
        # Example response
        return jsonify({
            "status": "success",
            "message": f"Backup task has been submitted."
        }), 200
    except Exception as e:
        # Catch unexpected errors and return a 500 status
        return jsonify({
            "status": "error",
            "message": f"An error occurred while initiating backup: {str(e)}"
        }), 500