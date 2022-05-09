import typing as tp

from .constants import STRING, INTEGER, PLAIN, NEIGHBOR, RELATIVE, ABSOLUTE

ALLOWED_FEATURES = [STRING, INTEGER]
ALLOWED_EMBEDDINGS = [PLAIN, NEIGHBOR]
ALLOWED_POSITION_EMBEDDINGS = [RELATIVE, ABSOLUTE]

def check_feature_type(feature_type: str) -> None:
    if feature_type not in ALLOWED_FEATURES:
            raise ValueError(f"`feature_type` must be in {ALLOWED_FEATURES}, got {feature_type}")

def check_embedding_type(embedding_type: str) -> None:
    if embedding_type not in ALLOWED_EMBEDDINGS:
            raise ValueError(f"`feature_type` must be either {ALLOWED_EMBEDDINGS}, got {embedding_type}")

def check_position_embedding_type(position_embedding_type: str) -> None:
    if position_embedding_type not in ALLOWED_POSITION_EMBEDDINGS:
            raise ValueError(f"`feature_type` must be either {ALLOWED_POSITION_EMBEDDINGS}, got {position_embedding_type}")
