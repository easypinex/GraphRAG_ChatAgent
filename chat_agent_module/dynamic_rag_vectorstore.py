import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.graphs.graph_document import (Document)
from chat_agent_module.search_params import SearchParamsDict
from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector

def get_localsearch_vectorstore(embedding) -> TwlfNeo4jVector:
    lc_retrieval_query = """
        // MATCH (n:__Entity__) WITH collect(n) AS nodes return nodes
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)<-[:HAS_CHILD]->(p:__Parent__)
            WITH c, p, count(distinct n) as freq
            RETURN {content: p.content, source: p.source} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        // Entity - Report Mapping
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY*]->(c:__Community__)
            WHERE c.summary is not null
            WITH c, c.rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        // Outside Relationships 
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m) 
            WHERE NOT m IN nodes and r.description is not null
            RETURN {description: r.description, sources: r.sources} AS descriptionText
            ORDER BY r.rank DESC, r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        // Inside Relationships 
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m) 
            WHERE m IN nodes and r.description is not null
            RETURN {description: r.description, sources: r.sources} AS descriptionText
            ORDER BY r.rank DESC, r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        // Entities description
        collect {
            UNWIND nodes as n
            match (n)
            WHERE n.description is not null
            RETURN {description: n.description, sources: n.sources} AS descriptionText
        } as entities
        // We don't have covariates or claims here
        RETURN {Chunks: text_mapping, Reports: report_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities} AS text, 1.0 AS score, {} AS metadata
    """

    vectorstore = TwlfNeo4jVector.from_existing_graph(embedding=embedding, 
                                        index_name="embedding",
                                        node_label='__Entity__', 
                                        embedding_node_property='embedding', 
                                        text_node_properties=['id', 'description'],
                                        retrieval_query=lc_retrieval_query)
    return vectorstore

if __name__ == "__main__":
    grpah = Neo4jGraph()
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment='text-embedding-3-small',
        openai_api_version='2023-05-15'
    )
    vectorstore = get_localsearch_vectorstore(embedding)
    search_params = SearchParamsDict.default()
    search_params['fileIds'] = ["bbe9f0d5-a3a3-49c2-92e7-07367d14ebd1"]
    topEntities = 3
    search_result: list[tuple[Document, float]] = vectorstore.similarity_search_with_score("台灣人壽", 
                                                                                           params=search_params, 
                                                                                           k=topEntities,
                                                                                           additional_match_cypher="MATCH (n:__Entity__)<-[:HAS_ENTITY]-(c:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]->(d:__Document__)",
                                                                                           additional_where_cypher="WHERE $fileIds IS NULL OR d.id IN $fileIds")
    for doc, socre in search_result:
        print(doc.page_content)
        print('-' * 80)