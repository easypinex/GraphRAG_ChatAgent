from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
from models import Base
from urllib.parse import quote
import os

# 定義 pymssql 連接字串的參數
db_username = os.environ.get("DB_USERNAME", "sa")
db_password = os.environ.get("DB_PASSWORD", "Passw@rd")
db_server = os.environ.get("DB_SERVER", "localhost")  # 通常 MSSQL 預設埠號為 1433, 若有需要可加上 :1433
db_schema = os.environ.get("DB_SCHEMA", "ACP")

# 對密碼進行 URL 編碼
encoded_password = quote(db_password, safe='')

# 組合 SQLAlchemy 的連接字串（pymssql）
connection_string = f"mssql+pymssql://{db_username}:{encoded_password}@{db_server}/{db_schema}?charset=utf8"

# 建立 Engine
engine = create_engine(connection_string, echo=False)

# 建立 Session Factory
session_factory = sessionmaker(bind=engine)

# 建立 Scoped Session，以確保在多 thread 下有獨立的 session, 實現線程安全
db_session = scoped_session(session_factory)

def init_db(drop_table = False, create_table=True):
    # 匯入模型（在此匯入避免循環匯入問題）
    import models
    if drop_table:
        Base.metadata.drop_all(bind=engine)  # 清除所有表
    if create_table:
        Base.metadata.create_all(bind=engine)

def shutdown_session(exception=None):
    db_session.remove()


if __name__ == '__main__':
    init_db()
    result = db_session.execute(text("SELECT TOP 1 * FROM file_tasks"))
    row = result.fetchone()
    print(row)