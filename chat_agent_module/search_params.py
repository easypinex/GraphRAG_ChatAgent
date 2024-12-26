from typing import TypedDict


class SearchParamsDict(TypedDict):
    fileIds: list[str]
    topChunks: int
    topCommunities: int
    topOutsideRels: int
    topInsideRels: int
    filterLimit: int
    @staticmethod
    def default() -> 'SearchParamsDict':
        return {
            "fileIds": None,
            "topChunks": 3,
            "topCommunities": 3,
            "topOutsideRels": 10,
            "topInsideRels": 10,
            "filterLimit": 10
        }