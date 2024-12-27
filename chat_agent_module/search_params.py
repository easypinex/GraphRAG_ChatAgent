from typing import TypedDict


class LocalSearchParamsDict(TypedDict):
    fileIds: list[str]
    topChunks: int
    topCommunities: int
    topOutsideRels: int
    topInsideRels: int
    @staticmethod
    def default() -> 'LocalSearchParamsDict':
        return {
            "fileIds": None,
            "topChunks": 3,
            "topCommunities": 3,
            "topOutsideRels": 10,
            "topInsideRels": 10,
        }