from dataclasses import dataclass, field

from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node)

from neo4j_module.neo4j_object_serialization import dict_to_doc, dict_to_graph_document, dict_to_node, doc_to_dict, graph_document_to_dict, node_to_dict

    
@dataclass
class DocChunk:
    chunk_id: str
    chunk_doc: Document
    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "chunk_doc": doc_to_dict(self.chunk_doc)
        }

    @staticmethod
    def from_dict(data: dict):
        return DocChunk(
            chunk_id = data["chunk_id"],
            chunk_doc = dict_to_doc(data["chunk_doc"])
        )

@dataclass
class GraphGraphDetails:
    root_document: Document
    root_node: Node
    root_graph_document: GraphDocument
    
    def to_dict(self):
        return {
            "root_document": doc_to_dict(self.root_document),
            "root_node": node_to_dict(self.root_node),
            "root_graph_document": graph_document_to_dict(self.root_graph_document)
        }
    def from_dict(data: dict):
        return GraphGraphDetails(
            root_document = dict_to_doc(data["root_document"]),
            root_node = dict_to_node(data["root_node"]),
            root_graph_document = dict_to_graph_document(data["root_graph_document"])
        )

@dataclass
class SimpleGraph:
    file_id: int = 0
    chunks: list[DocChunk] = field(default_factory=list)
    details: list[GraphGraphDetails] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "file_id": self.file_id,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "details": [detail.to_dict() for detail in self.details]
        }
    
    def from_dict(data: dict):
        return SimpleGraph(
            file_id=data.get("file_id"),
            chunks=[DocChunk.from_dict(chunk) for chunk in data["chunks"]],
            details=[GraphGraphDetails.from_dict(detail) for detail in data["details"]]
        )

if __name__ == "__main__":
    import json
    doc = Document(page_content="page_content")
    doc_chunk: DocChunk = DocChunk(chunk_id="chunk_id", chunk_doc=doc)
    print(doc_chunk.to_dict())
    detail = GraphGraphDetails(root_document=Document(page_content="page_content"), 
                               root_node=Node(id="id"), 
                               root_graph_document=GraphDocument(nodes=[],
                                                                 relationships=[], 
                                                                 source=doc))
    build_result = SimpleGraph(chunks=[doc_chunk], details=[detail])
    json_str = json.dumps(build_result.to_dict(), indent=2)
    recover_result = SimpleGraph.from_dict(json.loads(json_str))
    print(recover_result)










