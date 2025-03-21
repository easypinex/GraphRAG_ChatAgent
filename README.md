## envriment:
   * 請複製 .env.sample 為 .env
   * 設置所有變數
   * Python 3.12 版 (或以上-自行測試)

## 啟動應用
   * 啟動完整應用前，應備妥 Minio、RabbitMQ、MSSQL、Neo4j等週邊服務
   * 若只是要開發，可以快速 至 simulated_env/* 底下 啟動相關周邊服務(需搭配docker-compose)
   * 啟動服務指令:
      ```sh
      dotenv run -- python chat_agent.py > chat_agent.log 2>&1 & 
      dotenv run -- gunicorn -c gunicorn.conf.py app:app > gunicorn.log 2>&1 &
      dotenv run -- python rabbitmq_consumer.py > rabbitmq_consumer.log 2>&1
      ```
      - 背景執行版本
      ```sh
      nohup dotenv run -- python chat_agent.py > chat_agent.log 2>&1 & nohup dotenv run -- gunicorn -c gunicorn.conf.py app:app > gunicorn.log 2>&1 & nohup dotenv run -- python rabbitmq_consumer.py > rabbitmq_consumer.log 2>&1 &
      ```

## 理解資料流
   * 若您想要理解整個應用的資料流，推薦首先參考 dataflow_module/dataflow_service.py 的 main function, 或是該檔案從頭看到尾也可以(一開始不建議觀看其他檔案呼叫方法!)
   * 接著搭配 test/test_data/serialization/*.json 了解資料結構, 對整體流程了解會有很大的幫助!
   * 對於任務流程， rabbitmq_consumer.py 進行很好的任務接收概述!
   * 大致架構參考資料: [Implementing ‘From Local to Global’ GraphRAG with Neo4j and LangChain: Constructing the Graph](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/)、[Integrating Microsoft GraphRAG into Neo4j](https://neo4j.com/developer-blog/microsoft-graphrag-neo4j/)

## 理解詢問(查詢)
   * 對於理解問題如何查詢資料庫以及如何與LLM進行合作運行聊天應用, 請參考 chat_agent.py

## 了解如何呼叫進行聊天
   * 注意！串接聊天模型需要對 SSE(Server-Sent Events) 事件流 有基本的概念與了解
   * 前端人員要了解如何呼叫聊天應用 請參考 chat_agent_client_sample.py



## 資料庫備份
```sh
mkdir backup
docker run --rm \
  -v ./neo4j/data:/data \
  -v ./backup:/backup \
  neo4j:5.22.0 \
  neo4j-admin database dump neo4j --to-path=/backup
```

## 資料庫還原

```sh
docker run --rm \
  -v ./neo4j/data:/data \
  -v ./backup:/backup \
  neo4j:5.22.0 \
  neo4j-admin database load neo4j --from-path=/backup --overwrite-destination
```