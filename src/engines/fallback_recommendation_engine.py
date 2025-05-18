from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from typing import List, Dict, Any
from ..models.student_profile import StudentProfile
from config import get_settings
import json
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

SPECIALIZATION_PROMPT = """You are a university specialization advisor. Your task is to analyze the student's profile and provide 2-3 university specialization recommendations in JSON format.

Student Profile:
- Name: {name}
- Age: {age}
- Academic Level: {academic_level}
- Subjects: {subjects}
- Interests: {interests}
- Certifications: {certifications}
- Extracurriculars: {extracurriculars}
- Career Inclinations: {career_inclinations}
- Strengths: {strengths}
- Challenges: {challenges}

Instructions:
1. Focus on subjects they perform well in and enjoy
2. Consider their stated interests and career inclinations
3. Account for their academic level and challenges
4. Suggest specializations that align with their extracurricular activities

You must respond ONLY with a valid JSON array in the following format:
[
    {{
        "specialization": "Computer Science and Engineering",
        "reasoning": "Strong foundation in mathematics and computer science, demonstrated interest in programming through certifications, active participation in robotics club shows practical application skills",
        "key_subjects": ["Mathematics", "Computer Science", "Physics"],
        "career_prospects": ["Software Engineer", "Data Scientist", "Systems Architect"]
    }},
    {{
        "specialization": "Data Science",
        "reasoning": "Excellence in mathematics combined with programming skills and data science certifications shows natural alignment",
        "key_subjects": ["Statistics", "Computer Science", "Mathematics"],
        "career_prospects": ["Data Analyst", "Machine Learning Engineer", "Business Intelligence Analyst"]
    }}
]

Ensure your response contains ONLY the JSON array with no additional text or formatting."""

class FallbackRecommendationEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.7,  # Slightly increased for more creative recommendations
            api_key=settings.OPENAI_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_template(SPECIALIZATION_PROMPT)
        
        # Create a RunnableSequence chain
        self.chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def _format_profile(self, profile: StudentProfile) -> Dict[str, Any]:
        """Format student profile for the prompt template"""
        subjects_str = ", ".join([
            f"{s.name} ({s.grade or 'No grade'}, {'Favorite' if s.is_favorite else 'Not favorite'})"
            for s in profile.subjects
        ])
        
        return {
            "name": profile.name,
            "age": profile.age or "Not specified",
            "academic_level": profile.academic_level,
            "subjects": subjects_str,
            "interests": ", ".join(profile.interests) if profile.interests else "None specified",
            "certifications": ", ".join(profile.certifications) if profile.certifications else "None",
            "extracurriculars": ", ".join(profile.extracurriculars) if profile.extracurriculars else "None",
            "career_inclinations": ", ".join(profile.career_inclinations) if profile.career_inclinations else "None",
            "strengths": ", ".join(profile.strengths) if profile.strengths else "None",
            "challenges": ", ".join(profile.challenges) if profile.challenges else "None"
        }
    
    async def generate_recommendations(self, profile: StudentProfile) -> List[Dict]:
        """Generate specialization recommendations based on student profile"""
        try:
            # Format profile and generate recommendations
            variables = self._format_profile(profile)
            result = await self.chain.ainvoke(variables)
            
            # Clean the result string to ensure it only contains JSON
            result = result.strip()
            if not result.startswith('['):
                logger.error(f"Invalid JSON response format. Response: {result[:100]}...")
                raise ValueError("Response does not start with a JSON array")
            
            # Parse and validate JSON response
            try:
                recommendations = json.loads(result)
                if not isinstance(recommendations, list):
                    raise ValueError("Recommendations must be a list")
                
                # Validate each recommendation
                for rec in recommendations:
                    required_fields = ["specialization", "reasoning", "key_subjects", "career_prospects"]
                    missing_fields = [field for field in required_fields if field not in rec]
                    if missing_fields:
                        raise ValueError(f"Missing required fields in recommendation: {missing_fields}")
                    
                    # Validate field types
                    if not isinstance(rec["key_subjects"], list) or not isinstance(rec["career_prospects"], list):
                        raise ValueError("key_subjects and career_prospects must be lists")
                
                return recommendations
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse recommendations JSON: {e}\nResponse: {result[:200]}...")
                raise
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [{
                "specialization": "Computer Science",
                "reasoning": "Based on your strong interest in mathematics and programming, plus involvement in robotics club. This is a fallback recommendation due to processing issues.",
                "key_subjects": ["Mathematics", "Computer Science", "Physics"],
                "career_prospects": ["Software Engineer", "Data Scientist", "Systems Engineer"]
            }] 