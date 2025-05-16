from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum

class AcademicLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"

class Subject(BaseModel):
    name: str
    grade: Optional[str] = None
    level: Optional[str] = None
    is_favorite: bool = False

class StudentProfile(BaseModel):
    name: str = Field(..., min_length=2)
    age: Optional[int] = Field(None, ge=14, le=100)
    academic_level: AcademicLevel
    subjects: List[Subject] = Field(..., min_items=1)
    certifications: Optional[List[str]] = Field(default_factory=list)
    interests: List[str] = Field(..., min_items=1)
    extracurriculars: Optional[List[str]] = Field(default_factory=list)
    career_inclinations: Optional[List[str]] = Field(default_factory=list)
    strengths: Optional[List[str]] = Field(default_factory=list)
    challenges: Optional[List[str]] = Field(default_factory=list)
    
    @validator('subjects')
    def validate_subjects(cls, v):
        if not any(subject.is_favorite for subject in v):
            raise ValueError("At least one subject must be marked as favorite")
        return v
    
    def completion_percentage(self) -> float:
        """Calculate profile completion percentage based on filled fields"""
        total_fields = 10  # Total number of possible fields
        filled_fields = sum([
            bool(self.name),
            bool(self.age),
            bool(self.academic_level),
            bool(self.subjects),
            bool(self.certifications),
            bool(self.interests),
            bool(self.extracurriculars),
            bool(self.career_inclinations),
            bool(self.strengths),
            bool(self.challenges)
        ])
        return (filled_fields / total_fields) * 100

    def get_missing_fields(self) -> List[str]:
        """Return list of missing required fields"""
        missing = []
        if not self.name:
            missing.append("name")
        if not self.academic_level:
            missing.append("academic_level")
        if not self.subjects:
            missing.append("subjects")
        if not self.interests:
            missing.append("interests")
        return missing

    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "age": 17,
                "academic_level": "high_school",
                "subjects": [
                    {"name": "Mathematics", "grade": "A", "level": "Advanced", "is_favorite": True},
                    {"name": "Physics", "grade": "B+", "level": "Standard", "is_favorite": False}
                ],
                "interests": ["Programming", "Robotics"],
                "certifications": ["Python Programming"],
                "extracurriculars": ["Robotics Club"],
                "career_inclinations": ["Software Engineering"],
                "strengths": ["Problem Solving", "Analytical Thinking"],
                "challenges": ["Public Speaking"]
            }
        } 