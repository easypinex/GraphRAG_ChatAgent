import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from dataflow_module.rabbitmq_task import QueueTaskDict
import json

from flask import request, jsonify, Blueprint, current_app
from models.file_task import FileTask
from dto.file_upload_success import FileUploadSuccessDict
from dto.flask_error import FlaskErrorDict
from dataflow_module.rabbitmq_sender import publish_queue_message
from dataflow_module.dataflow_service import dataflow_manager_instance
from database import db_session

file_module = Blueprint('file_module', __name__)

@file_module.route('/file', methods=['POST'])
def upload_file():
    # 檢查是否有文件上傳
    if 'file' not in request.files:
        return jsonify(FlaskErrorDict(error = "No file part in the request")), 400
    file = request.files['file']
    # 如果文件名稱為空
    if file.filename == '':
        return jsonify(FlaskErrorDict(error = "No selected file")), 400
    # 保存文件
    UPLOAD_FOLDER = os.environ.get('USER_UPLOAD_TMP_FOLDER', 'upload_files')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    new_file_task = FileTask(filedir = UPLOAD_FOLDER, filename=file.filename)
    db_session.add(new_file_task)
    db_session.commit()
    queue_task = QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.FILE_READ, msg=json.dumps(new_file_task.to_dict()))
    publish_queue_message(queue_task)
    return jsonify(FileUploadSuccessDict(message = "File uploaded successfully", file_id=new_file_task.id))

@file_module.route('/file/<int:file_id>', methods=['GET'])
def get_status(file_id: int):
    file_task: FileTask = db_session.query(FileTask).get(file_id)
    if file_task is None:
        return jsonify(FlaskErrorDict(error = "File task not found")), 404
    return jsonify(file_task.to_dict())

@file_module.route('/file/<int:file_id>', methods=['DELETE'])
def delete_file(file_id: int):
    file_task: FileTask = db_session.query(FileTask).get(file_id)
    if file_task is None:
        return jsonify(FlaskErrorDict(error = "File task not found")), 404
    if file_task.status == FileTask.FileStatus.GRAPH_PROCESSING:
        return jsonify(FlaskErrorDict(error = "System Processing.. Please try late...")), 400
    dataflow_manager_instance._knowledge_service.remove_document_chain(file_id)
    dataflow_manager_instance._knowledge_service.remove_standalone_node()
    file_task.user_operate = FileTask.UserOperate.DELETE
    db_session.commit()
    return jsonify(FileUploadSuccessDict(message = "File Delete successfully", file_id=file_id))

    