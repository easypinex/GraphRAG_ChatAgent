from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
# 提供 Neo4j 原生物件序列化和反序列化的函数
# 序列化成 dict
def node_to_dict(node: Node, include_properties: bool = True) -> dict:
    result = {
            "id": node.id,
            "type": node.type,
            "properties": {}
        }
    if not include_properties:
        return result
    result.update({
        "properties": node.properties,
    })
    return result
def rel_to_dict(rel: Relationship) -> dict:
    return {
        "source": node_to_dict(rel.source, include_properties=False),
        "target": node_to_dict(rel.target, include_properties=False),
        "type": rel.type,
        "properties": rel.properties
    }
def doc_to_dict(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }
def graph_document_to_dict(graph_document: GraphDocument) -> dict:
    return {
        "nodes": [node_to_dict(node) for node in graph_document.nodes],
        "relationships": [rel_to_dict(rel) for rel in graph_document.relationships],
        "source": doc_to_dict(graph_document.source)
    }
    
# dict 反序列化為物件
def dict_to_node(data: dict) -> Node:
    return Node(id=data["id"], properties=data.get("properties", {}), type=data["type"])

def dict_to_rel(data: dict) -> Relationship:
    return Relationship(source=dict_to_node(data["source"]), 
                        target=dict_to_node(data["target"]), 
                        type=data["type"], 
                        properties=data["properties"])
def dict_to_doc(data: dict) -> Document:
    return Document(page_content=data["page_content"], metadata=data["metadata"])

def dict_to_graph_document(data: dict) -> GraphDocument:
    return GraphDocument(
        nodes=[dict_to_node(node) for node in data["nodes"]],
        relationships=[dict_to_rel(rel) for rel in data["relationships"]],
        source=dict_to_doc(data["source"])
    )