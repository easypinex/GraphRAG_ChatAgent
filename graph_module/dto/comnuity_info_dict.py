from typing import TypedDict
from collections import Counter

class Neo4jCommunityInfoDict(TypedDict):
    '''
    example:
    {
        "communityId": "0-1",
        "nodes": [
            {
                "id": "保險金給付之限制",
                "description": "被保險人已獲得全民健康保險給付的部分，本公司不予給付保險金。",
                "type": "概念",
                "sources": ["台灣人壽新住院醫療保險附約.pdf"],
                "uuid": "uuid()"
            },
            {"id": "第十四條", "type": "條款", "sources": ["台灣人壽新住院醫療保險附約.pdf"], "uuid": "uuid()"},
        ],
        "rels": [{"start": "第十四條", "description": None, "type": "描述", "end": "保險金給付之限制", "uuid": "uuid()"}],
    }

    '''
    communityId: str
    nodes: list[dict]
    rels: list[dict]

def compare_community_lists(listA: list[Neo4jCommunityInfoDict], listB: list[Neo4jCommunityInfoDict]) -> bool:
    def extract_features(community: Neo4jCommunityInfoDict):
        # 從 community 中提取 node_uuids 與 rel_uuids 的集合
        node_uuids = frozenset(node["uuid"] for node in community["nodes"])
        return (node_uuids)

    # 將 listA 中的 community 特徵收集
    featuresA = Counter(extract_features(c) for c in listA)
    # 將 listB 中的 community 特徵收集
    featuresB = Counter(extract_features(c) for c in listB)

    # 如果兩者的特徵分佈相同，則代表 listA 與 listB 社群完全一樣
    if featuresA == featuresB:
        # 兩者特徵完全匹配
        return True, []

    # 找出不一致部分
    # 在 A 裡有但在 B 裡不夠(或沒有)的特徵
    a_extra = (featuresA - featuresB)
    # 在 B 裡有但在 A 裡不夠(或沒有)的特徵
    b_extra = (featuresB - featuresA)

    differences = {
        'listA_size': len(listA),
        'listB_size': len(listB),
        'in_listA_not_in_listB_size': len(a_extra),
        'in_listB_not_in_listA_size': len(b_extra),
        'in_listA_not_in_listB': [],
        'in_listB_not_in_listA': []
    }

    # 從 listA 找出 a_extra 中對應的社群
    if a_extra:
        for community in listA:
            feat = extract_features(community)
            # 若此社群的特徵在 a_extra 中出現
            if feat in a_extra:
                differences['in_listA_not_in_listB'].append({
                    "source": "listA",
                    "communityId": community["communityId"],
                    "nodes": community["nodes"],
                    "rels": community["rels"]
                })

    # 從 listB 找出 b_extra 中對應的社群
    if b_extra:
        for community in listB:
            feat = extract_features(community)
            # 若此社群的特徵在 b_extra 中出現
            if feat in b_extra:
                differences['in_listB_not_in_listA'].append({
                    "source": "listB",
                    "communityId": community["communityId"],
                    "nodes": community["nodes"],
                    "rels": community["rels"]
                })
    def sort_func(x: list):
        for y in x:
            y['nodes'] = sorted(y['nodes'], key=lambda z: z['uuid'])
            y['rels'] = sorted(y['rels'], key=lambda z: z['uuid'])
    sort_func(differences['in_listA_not_in_listB'])
    sort_func(differences['in_listB_not_in_listA'])
    return False, differences


if __name__ == "__main__":
    # 範例使用:
    # 假設有以下兩個 list，裡面分別有 community 資訊
    listA = [
        {
            "communityId": "0-1",
            "nodes": [
                {"id": "保險金給付之限制", "description": "...", "type": "概念", "sources": ["xxx.pdf"], "uuid": "uuid-node-1"},
                {"id": "第十四條", "type": "條款", "sources": ["xxx.pdf"], "uuid": "uuid-node-2"},
            ],
            "rels": [
                {"start": "第十四條", "description": None, "type": "描述", "end": "保險金給付之限制", "uuid": "uuid-rel-1"}
            ]
        }
    ]

    listB = [
        {
            "communityId": "9-9",
            "nodes": [
                {"id": "保險金給付之限制", "description": "...", "type": "概念", "sources": ["xxx.pdf"], "uuid": "uuid-node-1"},
                {"id": "第十四條", "type": "條款", "sources": ["xxx.pdf"], "uuid": "uuid-node-2"},
            ],
            "rels": [
                {"start": "第十四條", "description": None, "type": "描述", "end": "保險金給付之限制", "uuid": "uuid-rel-1"}
            ]
        }
    ]

    # 即使 communityId 不同，但其他 uuid 都相同，所以結果會是 True
    print(compare_community_lists(listA, listB))  # True