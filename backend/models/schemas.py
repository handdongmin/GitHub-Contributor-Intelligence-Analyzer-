from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class RepoRequest(BaseModel):
    owner: str
    repo: str
    since: Optional[datetime] = None
    until: Optional[datetime] = None


class UserReposRequest(BaseModel):
    username: str


class TextsPayload(BaseModel):
    texts: List[str]


class SummarizePayload(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

    @model_validator(mode="after")
    def ensure_text(self):
        if not self.text and not self.texts:
            raise ValueError("text or texts is required")
        return self


class ClusterPayload(TextsPayload):
    k: int = Field(default=3, ge=2, le=20)


class TopicPayload(TextsPayload):
    # num_topics None이면 서버가 자동으로 결정
    num_topics: Optional[int] = Field(default=None, ge=2, le=30)


class ProductivityStat(BaseModel):
    commits: int
    issues: int
    additions: int
    deletions: int
    active_days: int
    prs: int = 0


class ProductivityPayload(BaseModel):
    stats: List[ProductivityStat]
    scores: List[float]
