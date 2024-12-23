import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, TypedDict

from graphdatascience import GraphDataScience 
from graph_module.dto.comnuity_info_dict import Neo4jCommunityInfoDict
from graph_module.dto.duplicate_info_dict import DuplicateInfoDict
from graph_module.dto.simple_graph import SimpleGraph
from graph_module.dto.summary_info_dict import SummaryInfoDict
from minio_module.minio_service import MinioService, minio_service
from models.file_task import FileTask
from neo4j_module.neo4j_object_serialization import dict_to_graph_document
from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from langchain_community.graphs.graph_document import (GraphDocument)
from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)

from prompts.prompts import COMMNUITY_SUMMARY_SYSTEM, PROMPT_FIND_DUPLICATE_NODES_SYSTEM, PROMPT_FIND_DUPLICATE_NODES_USER
import pandas as pd

class KnowledgeService:
    class PotentialDuplicateNodeDict(TypedDict):
        combinedResult: list[str]
    
    def __init__(self, graph, embedding, llm):
        self.graph = graph
        self.embedding = embedding
        self.llm = llm
        
    # 找出相似節點
    def findout_similar_nodes_rule_base(self) -> List[PotentialDuplicateNodeDict]:
        # 建立 Entity 所有 Embedding
        self.embedding_entities()
        gds = GraphDataScience( 
            os.environ[ "NEO4J_URI" ],
            auth=(os.environ[ "NEO4J_USERNAME" ], os.environ[ "NEO4J_PASSWORD" ]) 
        )
        gds.graph.drop("entities")
        # 检查是否存在 __Entity__ 节点
        query = """
        MATCH (n:`__Entity__`)
        RETURN COUNT(*) AS count
        """
        result = self.graph.query(query)
        if result[0]["count"] == 0:
            print("No '__Entity__' nodes found in the database.")
            return []
        
        G, result = gds.graph.project(
            "entities",                   # Graph name
            "__Entity__",                 # Node projection
            "*",                          # Relationship projection
            nodeProperties=["embedding"]  # Configuration parameters
        )
        # 使用 gds.knn.mutate 根據嵌入向量相似度創建關聯關係
        similarity_threshold = 0.95
        gds.knn.mutate(
            G,
            nodeProperties=['embedding'],
            mutateRelationshipType= 'SIMILAR',
            mutateProperty= 'score',
            similarityCutoff=similarity_threshold
        )
        
        # 使用 gds.wcc.write 將相似節點進行社群劃分
        # writeProperty="wcc": 為每個節點寫入 wcc 屬性，表示該節點屬於哪個社群。

        gds.wcc.write(
            G,
            writeProperty="wcc",
            relationshipTypes=["SIMILAR"]
        )
        
        # 查找具有潛在重複 ID 的節點
        word_edit_distance = 3
        potential_duplicate_candidates = self.graph.query(
            """MATCH (e:`__Entity__`)
            WHERE size(e.id) > 3 // longer than 3 characters
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance
            WITH distinct
            [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                        OR node.id CONTAINS n.id | n.id] AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // combine groups together if they share elements
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                    CASE WHEN index <> index2 AND
                        size(apoc.coll.intersection(acc, results[index2])) > 0
                        THEN apoc.coll.union(acc, results[index2])
                        ELSE acc
                    END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // extra filtering
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """, params={'distance': word_edit_distance})
        return potential_duplicate_candidates
    
    def determine_similar_nodes_with_cached_llm(self, potential_duplicate_candidates: list[PotentialDuplicateNodeDict], 
                                                    cached_duplicate_nodes: list[DuplicateInfoDict]) -> list[DuplicateInfoDict]:
        if len(cached_duplicate_nodes) == 0:
            return self.determine_similar_nodes_with_llm(potential_duplicate_candidates)
        # 1. 建立快取索引：key 為 frozenset(cached_input)，value 為對應的 DuplicateInfoDict
        cache_map: dict[frozenset[str], DuplicateInfoDict] = {}
        for dup_info in cached_duplicate_nodes:
            key = frozenset(dup_info["input"])
            cache_map[key] = dup_info

        # 2. 準備要呼叫 determine_similar_nodes_with_llm 的新資料和已經可直接回傳的結果
        new_candidates: list[KnowledgeService.PotentialDuplicateNodeDict] = []
        result_from_cache: List[DuplicateInfoDict] = []

        for candidate in potential_duplicate_candidates:
            candidate_key = frozenset(candidate["combinedResult"])
            if candidate_key in cache_map:
                # 從快取中取結果
                result_from_cache.append(cache_map[candidate_key])
            else:
                # 無快取紀錄，需要呼叫LLM計算
                new_candidates.append(candidate)

        # 3. 對新資料呼叫 determine_similar_nodes_with_llm
        if len(new_candidates) > 0:
            new_results = self.determine_similar_nodes_with_llm(new_candidates)
        else:
            new_results = []

        # 4. 合併結果並回傳
        return result_from_cache + new_results
        
    # 透過llm確認相似節點是否真的相似
    def determine_similar_nodes_with_llm(self, potential_duplicate_candidates: list[PotentialDuplicateNodeDict]) -> list[DuplicateInfoDict]:
        '''
        potential_duplicate_candidates like :
        [       
            {'combinedResult': ['住院醫療費用保險金限額', '住院醫療費用保險金限額_2']},
            {'combinedResult': ['每日住院病房費用保險金限額', '每日住院病房費用保險金限額_2']},
            ...
        ]
        '''

        system_prompt = PROMPT_FIND_DUPLICATE_NODES_SYSTEM
        user_template = PROMPT_FIND_DUPLICATE_NODES_USER


        class DuplicateEntities(BaseModel):
            entities: List[str] = Field(
                description="Entities that represent the same object or real-world entity and should be merged"
            )


        class Disambiguate(BaseModel):
            merge_entities: Optional[List[DuplicateEntities]] = Field(
                description="Lists of entities that represent the same object or real-world entity and should be merged"
            )


        extraction_llm = self.llm.with_structured_output(
            Disambiguate
        )

        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    user_template,
                ),
            ]
        )

        extraction_chain = extraction_prompt | extraction_llm

        def resolve_and_merge_entities_with_llm(potential_duplicate_candidates, max_retry=0) -> list[DuplicateInfoDict]:
            '''
            parmas:
                potential_duplicate_candidates(List[dict['combinedResult': List[str]]): 有可能需要合併的清單 
                                                                                        e.g.[{'combinedResult': ['土地銀行', '第一銀行']}]
                max_retry: 最多嘗試次數, 假設為2, 則最多遞迴執行 2+1=3次
            return:
                merged_entities (List[dict['combinedResult': List[str]]) : LLM 確認過需要合併的清單
                                                                            e.g.[{'combinedResult': ['土地銀行', '第一銀行']}]
            '''
            def entity_resolution(entities: list[str]) -> DuplicateInfoDict:
                llm_result: Disambiguate = extraction_chain.invoke({"entities": entities})
                result = DuplicateInfoDict(input = entities, output = [])
                if llm_result.merge_entities is None:
                    return result
                result["output"] = [el.entities for el in llm_result.merge_entities]
                return result
                
            merged_entities_result: list[DuplicateInfoDict] = []
            merged_future_map = {}
            futures = []
            merged_failds = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submitting all tasks and creating a list of future objects
                for el in potential_duplicate_candidates:
                    future = executor.submit(entity_resolution, el['combinedResult'])
                    merged_future_map[future] = el
                    futures.append(future)
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing documents"
                ):
                    try:
                        to_merge: DuplicateInfoDict = future.result()
                        merged_entities_result.append(to_merge)
                    except Exception as e:
                        el = merged_future_map[future]
                        print(f'process element faild!:{el['combinedResult']}, error:\n{e}')
                        merged_failds.append(el)
            if len(merged_failds) > 0 and max_retry > 0:
                merged_entities_result.extend(resolve_and_merge_entities_with_llm(merged_failds, max_retry=max_retry-1))
            return merged_entities_result
        merged_entities = resolve_and_merge_entities_with_llm(potential_duplicate_candidates, max_retry=0)
        return merged_entities
    
    # 合併經確認的節點
    def merge_nodes(self, merged_entities: list[list[str]]):
        """
        將經過人工確認的多個 __Entity__ 節點合併為一個節點:
        1. 在合併前先取得所需屬性 (sources, uuid_hash, 以及合成 uuid 所需的基底值)。
        2. 使用 apoc.refactor.mergeNodes 合併節點。
        3. 使用排序後的舊節點 uuid 所組成的字串作為新節點的 uuid，保證同組合併節點後的 uuid 與 uuid_hash 穩定一致。
        """
        self.graph.query("""
        UNWIND $data AS candidates
        CALL {
            WITH candidates
            MATCH (e:__Entity__) WHERE e.id IN candidates
            WITH collect(e) AS nodes, collect(e.sources) AS allSources
            WITH nodes, apoc.coll.flatten(allSources) AS flatSources,
                // 計算新的 uuid_hash
                reduce(acc = 0, h IN [n IN nodes | n.uuid_hash] | (acc + h) % 2147483647) AS newUuidHash,
                // 將要合併的所有節點 uuid 取出並排序
                apoc.coll.sort([n IN nodes | n.uuid]) AS sortedUuids
            // 將排序後的uuid拼接成一個字串，使之成為新uuid的基底
            WITH nodes, flatSources, newUuidHash, apoc.util.md5(sortedUuids) AS combinedUuid

            // 合併節點
            CALL apoc.refactor.mergeNodes(nodes, {
                properties: {
                    description: 'combine',
                    `.*`: 'discard'
                }
            }) YIELD node

            // 設定合併後的節點屬性
            SET node.uuid = combinedUuid,
                node.uuid_hash = newUuidHash,
                node.sources = apoc.coll.toSet(flatSources),
                node.merged = true
            RETURN count(*) AS c
        }
        RETURN sum(c)
        """, params={"data": merged_entities})


    def build_community_from_neo4j(self) -> list[Neo4jCommunityInfoDict]:
        """
        1. 透過 GDS library 將 Neo4j Graph 中的 __Entity__ 節點投影到 memory 中
        2. 進行社群偵測 WCC 演算法
        3. 將計算後的「社群資訊」寫入每個 __Entity__ 節點的 communities 屬性中, 可能會有多個社群
        4. 建立 Community 節點, 並將 __Entity__ 節點與 Community 節點相連
        5. 計算並儲存 Community Rank, (包含的Chunk數量)
        6. 取得至少有兩個子節點(Entity)的社群(Community), 並回傳所有關聯

        Returns:
            list[Neo4jCommunityInfoDict]:  A list of dictionaries which contain the community info.
        """
        gds = GraphDataScience( 
            os.environ[ "NEO4J_URI" ], 
            auth=(os.environ[ "NEO4J_USERNAME" ], os.environ[ "NEO4J_PASSWORD" ]) 
        )
        # 查詢節點數據
        node_query = """
        MATCH (n:__Entity__)
        RETURN n.uuid_hash AS nodeId
        ORDER BY nodeId
        """
        nodes_result = self.graph.query(node_query)

        # 查詢關係數據
        relationship_query = """
        MATCH (n:__Entity__)-[r]-(m:__Chunk__)
        WITH n, m, type(r) AS relationshipType, COUNT(r) AS connectionCount
        RETURN n.uuid_hash AS sourceNodeId, 
            id(m) AS targetNodeId, 
            relationshipType, 
            connectionCount AS weight
        ORDER BY sourceNodeId, targetNodeId
        """
        relationships_result = self.graph.query(relationship_query)

        # 將結果轉換為 Pandas DataFrame
        nodes = pd.DataFrame(nodes_result)
        relationships = pd.DataFrame(relationships_result)
        gds.graph.drop("communities")
        G = gds.graph.construct(
            "communities",      # Graph name
            nodes,           # One or more dataframes containing node data
            relationships,    # One or more dataframes containing relationship data,
            undirected_relationship_types=["*"]  # Set relationships as undirected
        )
                
        # node_props = gds.graph.nodeProperties("communities")
        # print(node_props)
        wcc = gds.wcc.stats(G)
        # 進行 wcc 社群偵測/分群演算法
        # 將計算後的「社群資訊」寫入每個 __Entity__ 節點的 communities 屬性中, 可能會有多個社群
        # communities可能會是類似 [0, 10, 42] 這樣的清單：
        #   0 可能代表最高層社群編號
        #   10 可能代表在更細分後層級的社群編號
        #   42 代表再細分一層後所屬的社群編號
        communities_df = gds.leiden.stream(
            G,
            # writeProperty="communities",
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
            randomSeed=27,  # 設定隨機種子, 確保每次結果都一致
            concurrency=1, # 設定執行緒數量, 確保每次結果都一致
            maxLevels=4
        )
        # 假設結果包含 nodeId 和 communityId 字段
        # print(communities_df.head())

        write_query = """
        UNWIND $data AS row
        MATCH (e:__Entity__ {uuid_hash: row.nodeId})
        SET e.communities = COALESCE(e.communities, []) + row.intermediateCommunityIds
        """

        # 構造參數列表
        params = {
            "data": communities_df.to_dict("records")
        }

        # 執行查詢
        self.graph.query(write_query, params)

        
        # 建立 Community 節點, 並將 __Entity__ 節點與 Community 節點相連
        self.graph.query("""
        MATCH (e:`__Entity__`)
        UNWIND range(0, size(e.communities) - 1 , 1) AS index
        CALL {
        WITH e, index
        WITH e, index
        WHERE index = 0
        MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET c.level = index
        MERGE (e)-[:IN_COMMUNITY]->(c)
        RETURN count(*) AS count_0
        }
        CALL {
        WITH e, index
        WITH e, index
        WHERE index > 0
        MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET current.level = index
        MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
        ON CREATE SET previous.level = index - 1
        MERGE (previous)-[:IN_COMMUNITY]->(current)
        RETURN count(*) AS count_1
        }
        RETURN count(*)
        """)
        
        # 計算並儲存 Community Rank, （包含的Chunk數量)
        self.graph.query("""
        MATCH (c:__Community__) 
        WHERE c.rank is null
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:HAS_ENTITY]-(d:__Chunk__)
        WITH c, count(distinct d) AS rank
        SET c.rank = rank;
        """)
        
        # 儲存 Summary 的 weight (也就是儲存 chunk 數量)
        self.graph.query(
        """
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount"""
        )
        
        # 取得至少有兩個子節點(Entity)的社群(Community), 並回傳所有關聯
        communities_info = self.graph.query("""
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
            // 由於前面只有最大4個層級, 因此只準備建立前兩層與最後一層的總結
            WHERE c.level IN [0,1,3]
            WITH c, collect(e ) AS nodes
            WHERE size(nodes) > 1 and c.summary is null
            
            // 检查是否存在父社区拥有相同的节点
            WITH c, nodes
            WHERE NOT EXISTS {
                MATCH (c:__Community__)<-[:IN_COMMUNITY]-(parent:__Community__)
                MATCH (parent)<-[:IN_COMMUNITY*]-(pe:__Entity__)
                WITH parent, collect(pe) AS parent_nodes, nodes
                WHERE size(parent_nodes) = size(nodes)
            }

            CALL apoc.path.subgraphAll(nodes[0], {
                whitelistNodes:nodes
            })
            YIELD relationships
            RETURN c.id AS communityId, 
                [n in nodes | apoc.map.removeKeys(n{.*, type: [el in labels(n) WHERE el <> '__Entity__'][0]}, ['embedding', 'wcc', 'communities'])] AS nodes,
                [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description, uuid: r.uuid}] AS rels
            """)
        
        return communities_info
        
    def summarize_commnuities_with_cached(
        self, 
        communities_info: List[Neo4jCommunityInfoDict], 
        cached_commnuities_info: List[Neo4jCommunityInfoDict], 
        cached_summaries: List[SummaryInfoDict]
    ) -> List[SummaryInfoDict]:
        '''
        根據給定的社群資訊以及快取的社群結構與摘要來取得每個社群的摘要。

        此函式會嘗試使用已快取的社群結構與摘要，以避免重複計算。
        若發現目前的社群結構在快取中已有對應的摘要，則直接使用；
        否則透過 LLM 重新計算摘要。

        Args:
            communities_info (List[Neo4jCommunityInfoDict]): 
                待產生摘要的社群資訊清單，每個社群包含 communityId 與 nodes。
            cached_commnuities_info (List[Neo4jCommunityInfoDict]): 
                已快取的社群資訊清單，用於比對結構是否一致。
            cached_summaries (List[SummaryInfoDict]): 
                已快取的社群摘要清單，通過 communityId 連結對應的摘要。

        Returns:
            List[SummaryInfoDict]: 
                與 communities_info 對應的社群摘要清單，每個元素包含 "community" (communityId) 與 "summary" (摘要內容)。

        時間複雜度:
            O((M + N) * K)，其中
            M 為 communities_info 的社群數量、
            N 為 cached_commnuities_info 的社群數量、
            K 為平均每個社群的 nodes 數量。

        程式流程:
            1. 建立 cached_commnuities_info 與 cached_summaries 的查詢索引。
            2. 使用每個 cached_commnuity 的 nodes UUIDs 建立結構特徵值 (frozenset) 作為索引鍵。
            3. 對每個 communities_info 中的社群計算其 nodes 的結構特徵值。
            - 若在快取中找到摘要，則直接使用。
            - 否則將該社群加入待重新計算的清單。
            4. 使用 summarize_with_llm 函式計算未快取的社群摘要。
            5. 最後依照 communities_info 的順序輸出所有摘要。

        注意事項:
            - 假設社群 ID 唯一且不變，但實際結構比對僅依據 nodes UUID。
            - 若快取中有結構相同的社群但無摘要，則會重新計算摘要。
        '''
        if len(cached_commnuities_info) == 0 or len(cached_summaries) == 0:
            return self.summarize_with_llm(communities_info)
        # 建立 cached_summary_dict 以方便透過 communityId 查 summary, 格式:{id: 敘述}
        cached_summary_dict = {cs['community']: cs['summary'] for cs in cached_summaries}
        
        # 預先計算 cached communities 的結構 key 與其 summary
        # key為 (frozenset_of_node_uuids, frozenset_of_rel_uuids)
        cached_struct_to_summary = {}
        for c in cached_commnuities_info:
            node_uuids = frozenset(node['uuid'] for node in c['nodes'])
            summary = cached_summary_dict.get(c['communityId'], None)
            cached_struct_to_summary[node_uuids] = summary

        result_summaries = []
        communities_to_summarize = []

        # 對需要總結的 communities 逐一檢查
        for community in communities_info:
            community_id = community['communityId']
            current_node_uuids = frozenset(node['uuid'] for node in community['nodes'])

            # 嘗試從 cached 中取得 summary
            key = current_node_uuids
            if key in cached_struct_to_summary:
                summary = cached_struct_to_summary.get(key)
                # 有 cached summary 可以直接用
                result_summaries.append({
                    "community": community_id,
                    "summary": summary
                })
            else:
                # 沒有 cached summary，需要重新計算
                communities_to_summarize.append(community)

        # 對需要重新計算的 communities 呼叫 summarize
        if communities_to_summarize:
            new_summaries = self.summarize_with_llm(communities_to_summarize)
            # 加入結果
            result_summaries.extend(new_summaries)

        # 最後依照原先 communities_info 順序輸出
        result_summaries_dict = {r['community']: r['summary'] for r in result_summaries}
        final_result = []
        for community in communities_info:
            cid = community['communityId']
            final_result.append({
                "community": cid,
                "summary": result_summaries_dict[cid]
            })
        
        return final_result
    
    def summarize_with_llm(self, communities_info: list[Neo4jCommunityInfoDict]) -> list[SummaryInfoDict]:
        community_template = """
        {community_info}
        Summary:"""  # noqa: E501

        community_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    COMMNUITY_SUMMARY_SYSTEM,
                ),
                (
                    "system", "使用繁體中文回應"
                ),
                ("human", community_template),
            ]
        )

        community_chain = community_prompt | self.llm | StrOutputParser()

        def prepare_string(data):
            nodes_str = "Nodes are:\n"
            for node in data['nodes']:
                desc_list = []
                for key in node:
                    if key == 'uuid':
                        continue
                    if node[key] is not None:
                        desc_list.append(f"{key}: {node[key]}")
                nodes_str += ', '.join(desc_list)

            rels_str = "Relationships are:\n"
            for rel in data['rels']:
                start = rel['start']
                end = rel['end']
                rel_type = rel['type']
                if 'description' in rel and rel['description']:
                    description = f", description: {rel['description']}"
                else:
                    description = ""
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            return nodes_str + "\n" + rels_str

        def process_community(community):
            stringify_info = prepare_string(community)
            summary = community_chain.invoke({'community_info': stringify_info})
            return {"community": community['communityId'], "summary": summary}

        def process_community_with_llm(communities_info, max_retry=0):
            '''
            params:
                community_info: [ { 
                                    'communityId': str, 'nodes': [{'id': str, 'description': str|None, 'type': str}, ...], 
                                    'rels': [{'start': str, 'description': str|None, 'type': str, 'end': 'str}, ...]
                                },
                                ... ]
                max_retry: 最多嘗試次數, 假設為2, 則最多遞迴執行 2+1=3次
            '''
            summaries = []
            faild_communities = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_community, community): community for community in communities_info}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing communities"):
                    try:
                        summaries.append(future.result())
                    except Exception as e:
                        community = futures[future]
                        print(f'process community faild!:{community}, error:\n{e}')
                        faild_communities.append(community)
            if len(faild_communities) > 0 and max_retry > 0:
                time.sleep(30) # 防止超出限制的情境, 等待一段時間後再嘗試
                summaries.extend(process_community_with_llm(faild_communities, max_retry=max_retry-1))
            return summaries
        summaries = process_community_with_llm(communities_info, max_retry=3)
        return summaries

    def save_summary(self, summaries: list[SummaryInfoDict]):
        # 儲存總結
        self.graph.query("""
        UNWIND $data AS row
        MERGE (c:__Community__ {id:row.community})
        SET c.summary = row.summary
        """, params={"data": summaries})
        
    def combine_description_list(self, split_str="\n---\n"):
        # 將 Entity 的 description 如果是陣列, 轉成字串 (因為 Entity 可能有多個來源來自不同的Chunk描述)
        # 同時移除掉 embedding, 之後就可以重新 embedding
        self.graph.query(f'''
        MATCH (n:__Entity__)
        WHERE n.description IS NOT NULL and apoc.meta.cypher.type(n.description) <> 'STRING'
        REMOVE n.embedding
        SET n.description = apoc.text.join(n.description, "{split_str}")
        RETURN n.description
        ''')
        self.embedding_entities()
        
    def embedding_entities(self):
        TwlfNeo4jVector.from_existing_graph(
            self.embedding,
            index_name='embedding',
            node_label='__Entity__',
            text_node_properties=['id', 'description'],
            embedding_node_property='embedding'
        )
        
    def get_builded_task_graph(self, tasks: list[FileTask]) -> dict[FileTask, tuple[SimpleGraph, list[GraphDocument]]]:
        result: dict[FileTask, tuple[SimpleGraph, list[GraphDocument]]] = {}
        for task in tasks:
            if task.status in [FileTask.FileStatus.REFINED_PENDING, FileTask.FileStatus.REFINING_KNOWLEDGE, 
                                FileTask.FileStatus.SUMMARIZING, FileTask.FileStatus.COMPLETED] and task.user_operate is None:
                entity_graph_list: list[dict] = minio_service.download_user_uploaded_metadata_from_minio_as_dict(task, MinioService.USER_UPLOADED_METADATA_TYPE.ENTITY_GRAPH_LIST)
                simple_graph_dict: dict = minio_service.download_user_uploaded_metadata_from_minio_as_dict(task, MinioService.USER_UPLOADED_METADATA_TYPE.SIMPLE_GRAPH)
                graph_documents: list[GraphDocument] = [dict_to_graph_document(graph_document) for graph_document in entity_graph_list]
                simple_graph: SimpleGraph = SimpleGraph.from_dict(simple_graph_dict)
                result[task] = ((simple_graph, graph_documents))
        return result
    
    def remove_all_entities_and_commnuities(self):
        self.remove_all_communities()
        self.remove_all_entities()
        
    def remove_all_entities(self):
        self.graph.query("""
            MATCH (n:__Entity__)
            DETACH DELETE n
        """)

    def remove_all_communities(self):
        self.graph.query("""
            MATCH (n:__Community__)
            DETACH DELETE n
        """)
        
    def remove_all_data(self):
        self.graph.query("MATCH (n) DETACH DELETE n")
        
    def remove_document_chain(self, file_task_id):
        self.graph.query(f"""
        MATCH (d:__Document__ {{file_task_id: {file_task_id}}})
        OPTIONAL MATCH (d)-[]-(p:__Parent__)
        OPTIONAL MATCH (p)-[]-(c:__Chunk__)
        OPTIONAL MATCH (c)-[]-(e:__Entity__)
        OPTIONAL MATCH (e)-[]-(m:__Community__)

        // 將找到的 Entity 與 Community 節點標註 reset = true
        SET e.reset = true
        SET m.reset = true

        // 刪除 Document、Parent、Chunk 節點
        DETACH DELETE d, p, c
    """)
    
    def remove_standalone_node(self):
        self.graph.query(f"""MATCH (e:__Entity__)
            WHERE NOT (e)--(:__Chunk__)
            DETACH DELETE e""")
        self.graph.query(f"""MATCH (co:__Community__)
                        WHERE NOT ((co)--(:__Entity__) OR (co)--(:__Community__)) 
                            AND (co.summary IS NULL OR co.summary = "")
                        DETACH DELETE co""")
    
    def fetch_graph_data(self):
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.graph = $graph_name AND m.graph = $graph_name
        RETURN n AS nodes, r AS relationships
        """
        return self.graph.query(query)

    def get_all_graph(self) -> GraphDocument:
        # 查询所有节点与关系
        # MATCH (n)-[r]->(m) 返回所有有向关系：n为起始节点，m为目标节点，r为关系
        results = self.graph.query("MATCH (n)-[r]->(m) RETURN n, r, m")

        # 当然，如果您想要包括孤立节点（无关系的节点），还可以额外执行：
        # MATCH (n) RETURN n
        # 然后将这些节点整合进来。
        # 这里的例子中先假设所有节点至少存在一条关系。

        node_dict = {}
        relationship_list = []

        for record in results:
            n_node = record["n"]
            r_rel = record["r"]
            m_node = record["m"]
            
            # 从 n_node 中提取信息
            n_id = n_node['id']
            n_labels = n_node.get('labels', [])
            n_props = dict(n_node)  # 将 n_node 的属性转换为字典

            if n_id not in node_dict:
                node_dict[n_id] = Node(
                    id=n_id,
                    type=":".join(n_labels) if n_labels else "Node",
                    properties=n_props
                )

            # 从 m_node 中提取信息
            m_id = m_node['id']
            m_labels = m_node.get('labels', [])
            m_props = dict(m_node)
            
            if m_id not in node_dict:
                node_dict[m_id] = Node(
                    id=m_id,
                    type=":".join(m_labels) if m_labels else "Node",
                    properties=m_props
                )

            # 从 r_rel 中提取关系信息
            r_type = r_rel[1]
            r_props = dict(r_rel[2])
            relationship_list.append(
                Relationship(
                    source=node_dict[n_id],
                    target=node_dict[m_id],
                    type=r_type,
                    properties=r_props
                )
            )

        # 如有需要，额外查询没有连接关系的独立节点
        # 将其整合入 node_dict 中
        # isolated_results = self.graph.query("MATCH (n) WHERE NOT (n)--() RETURN n")
        # for record in isolated_results:
        #     iso_node = record["n"]
        #     iso_id = iso_node.id
        #     iso_labels = iso_node.labels
        #     iso_props = dict(iso_node)
        #     if iso_id not in node_dict:
        #         node_dict[iso_id] = Node(
        #             id=iso_id,
        #             type=":".join(iso_labels) if iso_labels else "Node",
        #             properties=iso_props
        #         )

        # 创建 GraphDocument
        graph_document = GraphDocument(
            nodes=list(node_dict.values()),
            relationships=relationship_list,
            source=Document(page_content="Graph Document", metadata={})
        )

        # 至此，graph_document 就包含了 Neo4j 中的所有节点和关系信息。
        return graph_document

    def compare_graph_documents(self, gd1: GraphDocument, gd2: GraphDocument):
        '''針對有uuid的節點與rel進行比較, 目前有的只有 Entity 之間的 Rel'''
        # 只針對有 uuid 的節點進行比較
        gd1_node_by_uuid = {node.properties['uuid']: node for node in gd1.nodes if 'uuid' in node.properties}
        gd2_node_by_uuid = {node.properties['uuid']: node for node in gd2.nodes if 'uuid' in node.properties}

        # 只針對有 uuid 的關係進行比較
        gd1_rel_by_uuid = {rel.properties['uuid']: rel for rel in gd1.relationships if 'uuid' in rel.properties}
        gd2_rel_by_uuid = {rel.properties['uuid']: rel for rel in gd2.relationships if 'uuid' in rel.properties}

        gd1_node_uuids = set(gd1_node_by_uuid.keys())
        gd2_node_uuids = set(gd2_node_by_uuid.keys())
        gd1_rel_uuids = set(gd1_rel_by_uuid.keys())
        gd2_rel_uuids = set(gd2_rel_by_uuid.keys())

        node_in_gd1_not_in_gd2 = gd1_node_uuids - gd2_node_uuids
        node_in_gd2_not_in_gd1 = gd2_node_uuids - gd1_node_uuids
        rel_in_gd1_not_in_gd2 = gd1_rel_uuids - gd2_rel_uuids
        rel_in_gd2_not_in_gd1 = gd2_rel_uuids - gd1_rel_uuids

        # 若無任何差異，代表完全相同，直接回傳 True
        if not node_in_gd1_not_in_gd2 and not node_in_gd2_not_in_gd1 and not rel_in_gd1_not_in_gd2 and not rel_in_gd2_not_in_gd1:
            return True

        # 若有差異，將差異以字典形式回傳
        differences = {
            "nodes_in_a_not_in_b": [gd1_node_by_uuid[uuid] for uuid in node_in_gd1_not_in_gd2],
            "nodes_in_b_not_in_a": [gd2_node_by_uuid[uuid] for uuid in node_in_gd2_not_in_gd1],
            "relationships_in_a_not_in_b": [gd1_rel_by_uuid[uuid] for uuid in rel_in_gd1_not_in_gd2],
            "relationships_in_b_not_in_a": [gd2_rel_by_uuid[uuid] for uuid in rel_in_gd2_not_in_gd1]
        }

        return differences