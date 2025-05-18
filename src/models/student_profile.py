from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
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

    @classmethod
    def from_dict_or_object(cls, value: Union[Dict, 'Subject']) -> 'Subject':
        """Create a Subject from either a dictionary or Subject object"""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(f"Cannot create Subject from {type(value)}")

    def __hash__(self):
        """Make Subject hashable by using its name as the hash"""
        return hash(self.name)

    def __eq__(self, other):
        """Define equality based on the subject name"""
        if isinstance(other, Subject):
            return self.name == other.name
        return False

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
    
    @validator('subjects', pre=True)
    def validate_subjects(cls, v):
        """Validate and convert subjects to proper Subject objects"""
        if isinstance(v, list):
            # Convert each item to a Subject object
            subjects = [
                Subject.from_dict_or_object(item) if not isinstance(item, Subject) else item
                for item in v
            ]
            # Ensure at least one subject is marked as favorite
            if not any(subject.is_favorite for subject in subjects):
                # Mark the first subject as favorite if none are marked
                if subjects:
                    subjects[0] = Subject(
                        name=subjects[0].name,
                        grade=subjects[0].grade,
                        level=subjects[0].level,
                        is_favorite=True
                    )
            return subjects
        raise ValueError("subjects must be a list")

    @validator('interests', 'certifications', 'extracurriculars', 'career_inclinations', 'strengths', 'challenges')
    def deduplicate_lists(cls, v):
        """Remove duplicates from list fields while preserving order"""
        if v:
            seen = set()
            deduped = []
            for item in v:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            return deduped
        return v

    def completion_percentage(self) -> float:
        """Calculate profile completion percentage based on filled fields"""
        total_fields = 10  # Total number of possible fields
        filled_fields = sum([
            bool(self.name and self.name not in ["Anonymous", "Anonymous Student"]),
            bool(self.age),
            bool(self.academic_level),
            bool(self.subjects and len(self.subjects) > 0),
            bool(self.certifications),
            bool(self.interests and len(self.interests) > 0),
            bool(self.extracurriculars),
            bool(self.career_inclinations),
            bool(self.strengths),
            bool(self.challenges)
        ])
        return (filled_fields / total_fields) * 100

    def get_missing_fields(self) -> List[str]:
        """Return list of missing required fields"""
        missing = []
        if not self.name or self.name in ["Anonymous", "Anonymous Student"]:
            missing.append("name")
        if not self.academic_level:
            missing.append("academic_level")
        if not self.subjects or len(self.subjects) == 0:
            missing.append("subjects")
        if not self.interests or len(self.interests) == 0:
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