import json
import logging
import requests
from typing import List, Dict, Any
from ..models.student_profile import StudentProfile
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

INSTRUCTION_TEXT = (
    "You are a career counselor. Analyze the student's profile and recommend suitable university specializations, "
    "with reasons and career prospects. Respond ONLY with a JSON array."
)

class RecommendationEngine:
    def __init__(self):
        self.api_url = settings.RECOMMENDATION_API_URL  # e.g., "https://your-api-url/generate"
        self.max_new_tokens = 128

    def _format_profile(self, profile: StudentProfile) -> str:
        """Convert student profile to a structured input text for the API"""
        subjects_text = []
        for subj in profile.subjects:
            fav = " and is a favorite subject" if subj.is_favorite else ""
            subjects_text.append(
                f"{subj.name} ({subj.level}) with a grade of {subj.grade}{fav}"
            )
        subjects_str = "; ".join(subjects_text)

        interests = ", ".join(profile.interests or [])
        certifications = ", ".join(profile.certifications or [])
        extracurriculars = ", ".join(profile.extracurriculars or [])
        strengths = ", ".join(profile.strengths or [])
        challenges = ", ".join(profile.challenges or [])
        career_inclinations = ", ".join(profile.career_inclinations or [])

        input_text = (
            f"{profile.name} is a {profile.age}-year-old {profile.academic_level.replace('_', ' ').title()} student. "
            f"They have studied subjects such as {subjects_str}. "
            f"Their interests include {interests}. "
            f"They hold certifications in {certifications}. "
            f"In addition, they have participated in extracurricular activities such as {extracurriculars}. "
            f"Some of their strengths are {strengths}, while they face challenges in {challenges}. "
            f"Their career inclinations include {career_inclinations}."
        )

        return input_text

    def generate_recommendations(self, profile: StudentProfile) -> List[Dict[str, Any]]:
        """Send student profile to external API and return parsed recommendations"""
        try:
            input_text = self._format_profile(profile)
            payload = {
                "instruction": INSTRUCTION_TEXT,
                "input_text": input_text,
                "max_new_tokens": self.max_new_tokens
            }

            response = requests.post(self.api_url, json=payload)
            if not response.ok:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                raise ValueError("API request failed")

            result = response.text.strip()
            if not result.startswith('['):
                logger.error(f"Invalid response format: {result[:100]}...")
                raise ValueError("Response does not start with a JSON array")

            recommendations = json.loads(result)
            if not isinstance(recommendations, list):
                raise ValueError("Recommendations must be a list")

            for rec in recommendations:
                required_fields = ["specialization", "reasoning", "key_subjects", "career_prospects"]
                missing = [field for field in required_fields if field not in rec]
                if missing:
                    raise ValueError(f"Missing required fields: {missing}")
                if not isinstance(rec["key_subjects"], list) or not isinstance(rec["career_prospects"], list):
                    raise ValueError("key_subjects and career_prospects must be lists")

            return recommendations

        except Exception as e:
            logger.error(f"Error during recommendation generation: {str(e)}")
            return [{
                "specialization": "Computer Science",
                "reasoning": "Fallback recommendation due to processing error. Based on strong performance in data science and software development.",
                "key_subjects": ["Computer Science", "Physics", "Mathematics"],
                "career_prospects": ["Software Engineer", "Data Scientist", "AI Specialist"]
            }]
