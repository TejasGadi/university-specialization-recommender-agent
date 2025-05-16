from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from ..models.student_profile import StudentProfile, Subject, AcademicLevel
import json
import logging
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class ExtractedSubject(BaseModel):
    """Pydantic model for extracted subject information"""
    name: str = Field(description="Name of the subject")
    grade: Optional[str] = Field(None, description="Grade in the subject if mentioned")
    level: Optional[str] = Field(None, description="Level of the subject if mentioned")
    is_favorite: bool = Field(description="Whether this is a favorite subject based on sentiment")

class ExtractedProfile(BaseModel):
    """Pydantic model for extracted profile information"""
    name: Optional[str] = Field(None, description="Student name if mentioned")
    age: Optional[int] = Field(None, description="Age if mentioned")
    academic_level: Optional[str] = Field(None, description="Academic level (high_school/undergraduate/graduate)")
    subjects: List[ExtractedSubject] = Field(default_factory=list, description="List of subjects with details")
    interests: List[str] = Field(default_factory=list, description="List of academic and personal interests")
    certifications: List[str] = Field(default_factory=list, description="List of certifications and courses")
    extracurriculars: List[str] = Field(default_factory=list, description="List of extracurricular activities")
    career_inclinations: List[str] = Field(default_factory=list, description="List of career interests and goals")
    strengths: List[str] = Field(default_factory=list, description="List of strengths and skills")
    challenges: List[str] = Field(default_factory=list, description="List of challenges or areas for improvement")

EXTRACTION_PROMPT = """You are an AI assistant helping to extract student profile information from their messages.
Based on the following message, extract relevant information about the student's academic profile.

Message: {message}

Current Profile Information:
{current_profile}

Important rules for extraction:
1. For subjects:
   - Create a separate entry for each subject mentioned
   - Mark is_favorite as true if:
     * The student explicitly says they "enjoy", "like", "love", or "prefer" the subject
     * The subject is mentioned with positive sentiment
     * The subject is mentioned in context of being their favorite or best subject
   - If no subjects are mentioned with positive sentiment, mark the first mentioned subject as favorite
   - ALWAYS include at least one subject

2. For interests:
   - Include both academic subjects they enjoy and any other mentioned interests
   - Convert subject preferences into related interests
   - Always include "problem-solving" and related skills when mentioned
   - ALWAYS include at least one interest (use subject interests if no others mentioned)

3. For academic level:
   - Must be one of: high_school, undergraduate, or graduate
   - Infer from context if not explicitly stated
   - Default to "high_school" if unclear

4. For certifications:
   - Include any mentioned courses, certifications, or training programs
   - Include the platform (e.g., Coursera, Udemy) if mentioned
   - Format as "Course Name (Platform)" if platform is mentioned

5. For extracurricular activities:
   - Include any clubs, sports, teams, or organizations mentioned
   - Include both academic and non-academic activities
   - Include leadership roles or positions if mentioned

6. For career inclinations:
   - Include any mentioned career paths, job roles, or professional interests
   - When someone mentions interest in a field (e.g., "interested in software engineering"), add it to career_inclinations
   - Include both specific roles and broader fields
   - Extract career goals from statements about what they enjoy doing professionally

7. For name:
   - Extract if mentioned
   - Must be at least 2 characters long
   - Use "Anonymous Student" if not provided

Return the extracted information in a structured format following the schema provided."""

class ProfileExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        
        # Create LLM with structured output
        self.structured_llm = self.llm.with_structured_output(ExtractedProfile)
        
        # Create the chain
        self.chain = self.prompt | self.structured_llm
    
    def _format_current_profile(self, profile: Optional[StudentProfile]) -> str:
        """Format current profile information for the prompt"""
        if not profile:
            return "No current profile information available."
        
        return f"""
        Name: {profile.name if hasattr(profile, 'name') else 'Not provided'}
        Age: {profile.age if hasattr(profile, 'age') else 'Not provided'}
        Academic Level: {profile.academic_level if hasattr(profile, 'academic_level') else 'Not provided'}
        Subjects: {', '.join(s.name for s in profile.subjects) if hasattr(profile, 'subjects') else 'None'}
        Interests: {', '.join(profile.interests) if hasattr(profile, 'interests') else 'None'}
        Certifications: {', '.join(profile.certifications) if hasattr(profile, 'certifications') else 'None'}
        Extracurriculars: {', '.join(profile.extracurriculars) if hasattr(profile, 'extracurriculars') else 'None'}
        Career Inclinations: {', '.join(profile.career_inclinations) if hasattr(profile, 'career_inclinations') else 'None'}
        Strengths: {', '.join(profile.strengths) if hasattr(profile, 'strengths') else 'None'}
        Challenges: {', '.join(profile.challenges) if hasattr(profile, 'challenges') else 'None'}
        """
    
    def _ensure_valid_profile(self, extracted_info: Dict) -> Dict:
        """Ensure the extracted profile meets validation requirements"""
        # Ensure name is valid
        if not extracted_info.get("name"):
            extracted_info["name"] = "Anonymous Student"
        
        # Ensure academic level is valid
        if not extracted_info.get("academic_level"):
            extracted_info["academic_level"] = AcademicLevel.HIGH_SCHOOL
        
        # Ensure at least one subject
        if not extracted_info["subjects"]:
            extracted_info["subjects"] = [Subject(
                name="General Studies",
                is_favorite=True
            )]
        
        # Ensure at least one favorite subject
        if not any(s.is_favorite for s in extracted_info["subjects"]):
            extracted_info["subjects"][0].is_favorite = True
        
        # Ensure at least one interest
        if not extracted_info["interests"]:
            # Use favorite subjects as interests
            favorite_subjects = [s.name for s in extracted_info["subjects"] if s.is_favorite]
            if favorite_subjects:
                extracted_info["interests"] = favorite_subjects
            else:
                extracted_info["interests"] = ["General Studies"]
        
        return extracted_info
    
    async def extract_profile_info(self, message: str, current_profile: Optional[StudentProfile] = None) -> Dict:
        """Extract profile information from user message"""
        try:
            # Format current profile for context
            current_profile_str = self._format_current_profile(current_profile)
            
            # Run the chain with structured output
            result = await self.chain.ainvoke({
                "message": message,
                "current_profile": current_profile_str
            })
            
            # Convert Pydantic model to dictionary
            extracted_info = result.model_dump()
            
            # Convert subjects to Subject objects
            if extracted_info["subjects"]:
                subjects = []
                for subject in extracted_info["subjects"]:
                    # Create Subject object
                    subject_obj = Subject(
                        name=subject["name"],
                        grade=subject.get("grade"),
                        level=subject.get("level"),
                        is_favorite=subject["is_favorite"]
                    )
                    subjects.append(subject_obj)
                    
                    # If subject is marked as favorite, add it to interests
                    if subject_obj.is_favorite:
                        if subject_obj.name not in extracted_info["interests"]:
                            extracted_info["interests"].append(subject_obj.name)
                
                extracted_info["subjects"] = subjects
            
            # Add problem-solving and coding to interests if mentioned
            if "problem-solving" in message.lower():
                if "Problem Solving" not in extracted_info["interests"]:
                    extracted_info["interests"].append("Problem Solving")
            if "coding" in message.lower():
                if "Coding" not in extracted_info["interests"]:
                    extracted_info["interests"].append("Coding")
            
            # Convert academic_level to enum if present
            if extracted_info.get("academic_level"):
                try:
                    extracted_info["academic_level"] = AcademicLevel(extracted_info["academic_level"])
                except ValueError:
                    extracted_info["academic_level"] = AcademicLevel.HIGH_SCHOOL
            
            # Ensure all validation requirements are met
            extracted_info = self._ensure_valid_profile(extracted_info)
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting profile information: {str(e)}")
            # Return a valid default profile that meets all requirements
            return {
                "name": "Anonymous Student",
                "academic_level": AcademicLevel.HIGH_SCHOOL,
                "subjects": [Subject(name="General Studies", is_favorite=True)],
                "interests": ["General Studies"],
                "certifications": [],
                "extracurriculars": [],
                "career_inclinations": [],
                "strengths": [],
                "challenges": []
            } 