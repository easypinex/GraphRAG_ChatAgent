from flask import Flask
from database import init_db, shutdown_session

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('flask_config.py')
    
    # 註冊 Blueprint
    with app.app_context():
        from file_module.file_flask_module import file_module
        app.register_blueprint(file_module, url_prefix='/file')
        from scheduler_module.scheduler_flask_module import scheduler_module
        app.register_blueprint(scheduler_module, url_prefix='/scheduler')
        from scheduler_module.scheduler_flask_module import init_scheduler
        # 初始化 Scheduler
        init_scheduler(app)
        
        from neo4j_module.neo4j_flask_module import neo4j_module
        app.register_blueprint(neo4j_module, url_prefix='/neo4j')
    
    return app

def flask_init_db(app):
    # 在這裡初始化 db
    init_db(drop_table=False, create_table=True)
    @app.teardown_appcontext
    def remove_session(exception=None):
        shutdown_session(exception)

app = create_app()
with app.app_context():
    # 使用 Base.metadata 操作資料庫
    flask_init_db(app)
    
if __name__ == '__main__':
    print(app.url_map)
    app.run()

