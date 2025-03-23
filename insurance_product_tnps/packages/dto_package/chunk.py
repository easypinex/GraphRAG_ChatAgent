
from dataclasses import dataclass, field

@dataclass
class Chunk:
    content: str = ""
    filename: str = ""
    summary: str = ""
    page: list[int] = field(default_factory=list)
    segment_list: list[str] = field(default_factory=list)
    topic_list: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "content": self.content,
            "filename": self.filename,
            "summary": self.summary,
            "page": self.page,
            "segment_list": self.segment_list,
            "topic_list": self.topic_list,
        }
    
    @staticmethod
    def from_dict(data: dict):
        return Chunk(
            content=data["content"],
            filename=data["filename"],
            summary=data["summary"],
            page=data["page"],
            segment_list=data["segment_list"],
            topic_list=data["topic_list"]
        )
    