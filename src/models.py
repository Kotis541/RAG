from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """A source chunk identified by file path and character range."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A dataset question without a ground-truth answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A dataset question with ground-truth sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Full RAG dataset containing answered and/or unanswered questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search result for a single question with retrieved source chunks."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result extended with a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Batch of student search results with the k parameter used."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Batch of student search results that also include generated answers."""
    search_results: List[MinimalAnswer]