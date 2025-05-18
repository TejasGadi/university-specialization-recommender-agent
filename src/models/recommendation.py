from pydantic import BaseModel, Field
from typing import List

class Specialization(BaseModel):
    """Model representing a single specialization recommendation"""
    specialization: str = Field(..., description="Name of the recommended specialization")
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    key_subjects: List[str] = Field(..., description="List of key subjects relevant to this specialization")
    career_prospects: List[str] = Field(..., description="List of potential career paths")
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the recommendation (0-1)"
    )

class RecommendationList(BaseModel):
    """Model representing a list of specialization recommendations"""
    recommendations: List[Specialization]
    generated_at: str = Field(..., description="Timestamp when recommendations were generated")
    profile_completion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score indicating how complete the student profile was"
    )
    feedback_incorporated: bool = Field(
        default=False,
        description="Whether previous user feedback was incorporated"
    ) 