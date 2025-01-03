'''
示例如何透過 LangServe, 提供之API stream_events, 獲取RAG資訊以及回應內容
'''

import uuid
import json
from copy import deepcopy

import sseclient
import requests

from datetime import datetime

# ! pip install sseclient-py

QUESTION = '台灣人壽'
URL = 'https://localhost:8000/stream_events'

input_json = {
    "input": {
        "question": QUESTION,
        "fileIds": []
    },
    "config": {
        "configurable": {
            "session_id": str(uuid.uuid4())
        }
    }
}

stream_response = requests.post(URL, json=input_json, stream=True, timeout=15, verify=False)

client = sseclient.SSEClient(stream_response)
for event in client.events():
    data = json.loads(event.data)
    name = data.get('name')
    event = data.get('event')
    tags = data.get('tags', [])
    in_data = data.get('data')
    file_datas = []
    # print(f'event: {event}, name: {name}, tags: {tags}, data:{data}')
    # print('-' * 40)
    if name == 'RunnableSequence':
        outputs: list[dict] = in_data.get('output') if in_data is not None and in_data.get('output') else []
        for output in outputs:
            if isinstance(output, dict):
                metadatas: list[dict] = [output.get('metadata') for output in outputs] if outputs is not None else []
                in_file_datas: list[list] = [metadata.get('file_datas') for metadata in metadatas if metadata.get('file_datas') is not None]
                for files in in_file_datas:
                    file_datas.extend(files)
    if file_datas:
        for document in file_datas:
            fileId = document['fileId']
            filename = document['filename']
            filelinke = document['filelink']
            print(f'fileId: {fileId}, filename: {filename}, filelink: {filelinke}')
            print('-' * 40)
            # content = document['page_content'].strip()
            # if content.startswith('content:'):
            #     content = content[len('content:'):]
            # content = content.strip()
            # metadata = deepcopy(document['metadata'])
            # if 'source' in metadata:
            #     del metadata['source']
            # print('內文: ' + content)
            # print('參數: ' + str(metadata))
            # print('-' * 40)
    elif event == 'on_parser_end' and 'contextualize_question' in tags:
        print('問題更新: ' + data['data']['output'])
        print('-' * 40)
    elif 'final_output' in tags:
        chunk = data['data'].get('chunk')
        if chunk is not None:
            print(chunk, end='', flush=True)
    # else:
    #     print(f'event: {event}, name: {name}, tags: {tags}, data:{data}')
    #     print('-' * 40)
print("")
print("-" * 40)
