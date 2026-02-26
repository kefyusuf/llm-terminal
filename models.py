from typing import TypedDict


class ModelResult(TypedDict):
    inst: str
    source: str
    provider: str
    id: str
    name: str
    params: str
    use_case: str
    score: str
    quant: str
    mode: str
    fit: str
    size: str
