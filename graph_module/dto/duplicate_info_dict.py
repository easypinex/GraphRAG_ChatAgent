from typing import TypedDict

class DuplicateInfoDict(TypedDict):
    """
    input: list[str], 可能需要合併的 Entity
    output: list[list[str]], LLM 覺得最終要合併的 Entity
    """
    input: list[str]
    output: list[list[str]]
    