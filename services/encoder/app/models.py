from redis_om import HashModel, Field


class Embedding(HashModel):
    """Embedded features for a single frame crop."""
    sm_name: str = Field(index=True)
    source_id: int = Field(index=True)
    timestamp: float = Field(index=True)
    features: str
