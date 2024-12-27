from typing import Any, List, Optional, Type
from langchain_core.embeddings import Embeddings
from langchain_neo4j.vectorstores.neo4j_vector import SearchType, \
                                                        DEFAULT_SEARCH_TYPE, Neo4jVector, \
                                                        construct_metadata_filter, \
                                                        IndexType, \
                                                        remove_lucene_chars, \
                                                        dict_to_yaml_str, \
                                                        DEFAULT_INDEX_TYPE
                                                        
from langchain_community.graphs.graph_document import (Document)
import overrides

class TwlfNeo4jVector(Neo4jVector):
    @classmethod
    def from_existing_graph(
        cls: 'TwlfNeo4jVector',
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> 'TwlfNeo4jVector':
        """
        參考 Neo4jVector.from_existing_graph 複寫 Neo4jVector.from_existing_graph 方法
        
        因為原本設置為每 1000筆 為一個批次, 可能單一批次就會造成 OpenAI Rate Limite 錯誤, 因此調整為 500 筆一個批次
        目前只有在 embedding Chunk Node 才發生此現象. Entity 跟 Commnuity 則正常
        
        以下為原生註解:
        --------------------------------------------------------------------
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null, id: Null, "
                + ", ".join([f"`{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata, score"
            )
        store = TwlfNeo4jVector(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists. "
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index(text_node_properties)
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            fetch_query = (
                f"MATCH (n:`{node_label}`) "
                f"WHERE n.{embedding_node_property} IS null "
                "AND any(k in $props WHERE n[k] IS NOT null) "
                f"RETURN elementId(n) AS id, reduce(str='',"
                "k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text "
                "LIMIT 500" # > 只調整這裡!! 1000 > 500
            )
            data = store.query(fetch_query, params={"props": text_node_properties})
            if not data:
                break
            text_embeddings = embedding.embed_documents([el["text"] for el in data])

            params = {
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE elementId(n) = row.id "
                f"CALL db.create.setNodeVectorProperty(n, "
                f"'{embedding_node_property}', row.embedding) "
                "RETURN count(*)",
                params=params,
            )
            # If embedding calculation should be stopped
            if len(data) < 500: # > 只調整這裡!! 1000 > 500
                break
        return store

    
    
    @overrides.overrides
    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        params: dict[str, Any] = {},
        effective_search_ratio: int = 1,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        覆寫原生 similarity_search_with_score_by_vector, 因為想要增加n額外關係, 以及額外關係的條件, 
        比如:
            n - [] - m, where m.propertis = ...
        透過 kwargs 來傳入: 
        {
            "additional_match_cypher": "MATCH ...",
            "additional_where_cypher": "WHERE ..."
        }
        以下為原生註解
        """
        """
        Perform a similarity search in the Neo4j database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the Neo4j database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            effective_search_ratio (int): Controls the candidate pool size
               by multiplying $k to balance query accuracy and performance.
               Defaults to 1.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        if filter:
            # Verify that 5.18 or later is used
            if not self.support_metadata_filter:
                raise ValueError(
                    "Metadata filtering is only supported in "
                    "Neo4j version 5.18 or greater"
                )
            # Metadata filtering and hybrid doesn't work
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )
            parallel_query = (
                "CYPHER runtime = parallel parallelRuntimeSupport=all "
                if self._is_enterprise
                else ""
            )
            base_index_query = parallel_query + (
                f"MATCH (n:`{self.node_label}`) WHERE "
                f"n.`{self.embedding_node_property}` IS NOT NULL AND "
                f"size(n.`{self.embedding_node_property}`) = "
                f"toInteger({self.embedding_dimension}) AND "
            )
            base_cosine_query = (
                " WITH n as node, vector.similarity.cosine("
                f"n.`{self.embedding_node_property}`, "
                "$embedding) AS score ORDER BY score DESC LIMIT toInteger($k) "
            )
            filter_snippets, filter_params = construct_metadata_filter(filter)
            index_query = base_index_query + filter_snippets + base_cosine_query

        else:
            index_query = _get_search_index_query(
                self.search_type, self._index_type, self.neo4j_version_is_5_23_or_above, **kwargs ## TWLF 新增 -> **kwargs, 並呼叫自己的 _get_search_index_query
            )
            filter_params = {}

        if self._index_type == IndexType.RELATIONSHIP:
            if kwargs.get("return_embeddings"):
                default_retrieval = (
                    f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                    f"relationship {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null, "
                    f"_embedding_: relationship.`{self.embedding_node_property}`}} "
                    "AS metadata"
                )
            else:
                default_retrieval = (
                    f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                    f"relationship {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
                )

        else:
            if kwargs.get("return_embeddings"):
                default_retrieval = (
                    f"RETURN node.`{self.text_node_property}` AS text, score, "
                    f"node {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null, "
                    f"_embedding_: node.`{self.embedding_node_property}`}} AS metadata"
                )
            else:
                default_retrieval = (
                    f"RETURN node.`{self.text_node_property}` AS text, score, "
                    f"node {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
                )

        retrieval_query = (
            self.retrieval_query if self.retrieval_query else default_retrieval
        )

        read_query = index_query + retrieval_query
        parameters = {
            "index": self.index_name,
            "k": k,
            "embedding": embedding,
            "keyword_index": self.keyword_index_name,
            "query": remove_lucene_chars(kwargs["query"]),
            "ef": effective_search_ratio,
            **params,
            **filter_params,
        }
        print(read_query, parameters)
        results = self.query(read_query, params=parameters)
        '''
        
        
        '''
        if any(result["text"] is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `text` column"
                )
        if kwargs.get("return_embeddings") and any(
            result["metadata"]["_embedding_"] is None for result in results
        ):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.embedding_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `_embedding_` metadata column"
                )

        docs = [
            (
                Document(
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]
        return docs

def _get_search_index_query(
    search_type: SearchType,
    index_type: IndexType = DEFAULT_INDEX_TYPE,
    neo4j_version_is_5_23_or_above: bool = False,
    **kwargs
) -> str:
    '''
    覆寫 Neo4j 的 _get_search_index_query, 用途為讀取 additional_match_cypher與 additional_where_cypher, 進行臨時篩選
    '''
    ## ---------- TWLF 新增 Start ----------
    additional_match_cypher = kwargs.get("additional_match_cypher", "") ## TWLF 新增
    additional_where_cypher = kwargs.get("additional_where_cypher", "") ## TWLF 新增
    if index_type == IndexType.NODE:
        if search_type == SearchType.VECTOR:
            additional_query = ""
            if additional_match_cypher or additional_where_cypher:
                additional_query =  (
                    f"{additional_match_cypher}"
                    f" {additional_where_cypher} "
                    " WITH collect(n) AS nodes "
                    " UNWIND nodes AS filteredNode "
                )
            return additional_query + (
                "CALL db.index.vector.queryNodes($index, $k * $ef, $embedding) "
                "YIELD node, score "
                "WITH node, score LIMIT $k "
            )
            ## ---------- TWLF 新增 End ----------
        elif search_type == SearchType.HYBRID:
            call_prefix = "CALL () { " if neo4j_version_is_5_23_or_above else "CALL { "

            query_body = (
                "CALL db.index.vector.queryNodes($index, $k * $ef, $embedding) "
                "YIELD node, score "
                "WITH node, score LIMIT $k "
                "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                "UNWIND nodes AS n "
                "RETURN n.node AS node, (n.score / max) AS score UNION "
                "CALL db.index.fulltext.queryNodes($keyword_index, $query, "
                "{limit: $k}) YIELD node, score "
                "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                "UNWIND nodes AS n "
                "RETURN n.node AS node, (n.score / max) AS score "
            )

            call_suffix = (
                "} WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
            )

            return call_prefix + query_body + call_suffix
        else:
            raise ValueError(f"Unsupported SearchType: {search_type}")
    else:
        return (
            "CALL db.index.vector.queryRelationships($index, $k * $ef, $embedding) "
            "YIELD relationship, score "
            "WITH relationship, score LIMIT $k "
        )