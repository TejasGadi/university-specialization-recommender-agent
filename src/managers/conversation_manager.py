import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from ..models.student_profile import StudentProfile
from ..utils.cache import Cache
from ..utils.rate_limiter import RateLimiter
from ..engines.recommendation_engine import RecommendationEngine
from ..engines.career_path_engine import CareerPathEngine
from ..extractors.profile_extractor import ProfileExtractor
from config import get_settings

settings = get_settings()

class ConversationState(BaseModel):
    stage: str = "welcome"  # welcome, profile_collection, recommendation, career_paths
    profile: Optional[StudentProfile] = None
    last_interaction: datetime = datetime.now()
    recommendations: Optional[list] = None
    selected_specialization: Optional[str] = None

class ConversationManager:
    def __init__(self):
        self.cache = Cache()
        self.rate_limiter = RateLimiter()
        self.states: Dict[str, ConversationState] = {}
        self.profile_extractor = ProfileExtractor()
        self.recommendation_engine = RecommendationEngine()
        self.career_path_engine = CareerPathEngine()
    
    async def process_message(self, session_id: str, message: str) -> str:
        """Process incoming message and return response within time constraint"""
        try:
            # Check rate limit
            await self.rate_limiter.check_rate_limit(session_id)
            
            # Get or create conversation state
            state = self.states.get(session_id, ConversationState())
            
            # Try to get cached response
            cache_key = f"{session_id}:{message}"
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                return cached_response
            
            # Process message with timeout
            response = await asyncio.wait_for(
                self._process_message_internal(state, message),
                timeout=settings.MAX_RESPONSE_TIME
            )
            
            # Update state
            state.last_interaction = datetime.now()
            self.states[session_id] = state
            
            # Cache response
            await self.cache.set(cache_key, response)
            
            return response
            
        except asyncio.TimeoutError:
            return "I need a moment to process your request. Could you please repeat or simplify your question?"
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"
    
    async def _process_message_internal(self, state: ConversationState, message: str) -> str:
        """Internal message processing logic based on conversation stage"""
        if state.stage == "welcome":
            state.stage = "profile_collection"
            return """Welcome! I'm your university specialization advisor. I'll help you find the best academic path 
                     based on your background and interests. Let's start by getting to know you. 
                     What subjects have you studied so far, and which ones do you enjoy most?"""
        
        elif state.stage == "profile_collection":
            # Update profile with new information
            updated_profile = await self._update_profile(state.profile, message)
            state.profile = updated_profile
            
            # Check if profile is complete
            if updated_profile.completion_percentage() >= 80:
                missing_fields = updated_profile.get_missing_fields()
                if not missing_fields:
                    state.stage = "recommendation"
                    return await self._generate_profile_summary(updated_profile)
            
            # Ask for missing information
            return await self._generate_next_question(updated_profile)
        
        elif state.stage == "recommendation":
            if "yes" in message.lower() or "correct" in message.lower():
                # Generate recommendations
                state.recommendations = await self.recommendation_engine.generate_recommendations(state.profile)
                state.stage = "career_paths"
                return await self._format_recommendations(state.recommendations)
            else:
                return "What information would you like to update in your profile?"
        
        elif state.stage == "career_paths":
            return await self._process_career_path_request(state, message)
    
    async def _update_profile(self, profile: Optional[StudentProfile], message: str) -> StudentProfile:
        """Update profile with information extracted from message"""
        # Extract new information from message
        extracted_info = await self.profile_extractor.extract_profile_info(message, profile)
        print(f"Extracted info in update_profile: {extracted_info}")  # Debug print
        
        if not profile:
            # Create new profile if none exists
            try:
                # Ensure minimum required fields
                if "subjects" not in extracted_info:
                    extracted_info["subjects"] = []
                if "interests" not in extracted_info:
                    extracted_info["interests"] = []
                if "name" not in extracted_info:
                    extracted_info["name"] = "Anonymous"
                if "academic_level" not in extracted_info:
                    extracted_info["academic_level"] = "high_school"
                if "certifications" not in extracted_info:
                    extracted_info["certifications"] = []
                if "extracurriculars" not in extracted_info:
                    extracted_info["extracurriculars"] = []
                if "career_inclinations" not in extracted_info:
                    extracted_info["career_inclinations"] = []
                if "strengths" not in extracted_info:
                    extracted_info["strengths"] = []
                
                return StudentProfile(**extracted_info)
            except Exception as e:
                print(f"Error creating profile: {str(e)}")
                # Create minimal valid profile
                return StudentProfile(
                    name="Anonymous",
                    academic_level="high_school",
                    subjects=[],
                    interests=[]
                )
        else:
            try:
                # Create a new dictionary with all existing profile data
                updated_info = profile.dict()
                print(f"Current profile before update: {updated_info}")  # Debug print
                
                # Update subjects if new ones are provided
                if extracted_info.get("subjects"):
                    updated_info["subjects"] = extracted_info["subjects"]
                
                # Update or append interests
                if extracted_info.get("interests"):
                    current_interests = set(updated_info.get("interests", []))
                    new_interests = set(extracted_info["interests"])
                    updated_info["interests"] = list(current_interests.union(new_interests))
                
                # Update or append certifications
                if extracted_info.get("certifications"):
                    current_certs = set(updated_info.get("certifications", []))
                    new_certs = set(extracted_info["certifications"])
                    updated_info["certifications"] = list(current_certs.union(new_certs))
                
                # Update or append extracurriculars
                if extracted_info.get("extracurriculars"):
                    current_extra = set(updated_info.get("extracurriculars", []))
                    new_extra = set(extracted_info["extracurriculars"])
                    updated_info["extracurriculars"] = list(current_extra.union(new_extra))
                
                # Update or append career inclinations
                if extracted_info.get("career_inclinations"):
                    current_careers = set(updated_info.get("career_inclinations", []))
                    new_careers = set(extracted_info["career_inclinations"])
                    updated_info["career_inclinations"] = list(current_careers.union(new_careers))
                    # Also add career inclinations to interests if not already there
                    current_interests = set(updated_info.get("interests", []))
                    updated_info["interests"] = list(current_interests.union(new_careers))
                
                # Update or append strengths
                if extracted_info.get("strengths"):
                    current_strengths = set(updated_info.get("strengths", []))
                    new_strengths = set(extracted_info["strengths"])
                    updated_info["strengths"] = list(current_strengths.union(new_strengths))
                
                print(f"Updated profile data: {updated_info}")  # Debug print
                
                # Create new profile with updated information
                return StudentProfile(**updated_info)
                
            except Exception as e:
                print(f"Error updating profile: {str(e)}")
                return profile
    
    async def _generate_next_question(self, profile: StudentProfile) -> str:
        """Generate next question based on missing profile information"""
        missing_fields = profile.get_missing_fields()
        
        if "name" in missing_fields:
            return "Could you tell me your name?"
        elif "academic_level" in missing_fields:
            return "Are you currently in high school, undergraduate, or graduate studies?"
        elif "subjects" in missing_fields:
            return "What subjects have you studied? Please include any grades if you'd like to share them."
        elif "interests" in missing_fields:
            return "What are your academic or personal interests?"
        
        # Track what we've already asked about to avoid loops
        has_subjects = len(profile.subjects) > 0
        has_certifications = profile.certifications and len(profile.certifications) > 0
        has_extracurriculars = profile.extracurriculars and len(profile.extracurriculars) > 0
        has_careers = profile.career_inclinations and len(profile.career_inclinations) > 0
        has_strengths = profile.strengths and len(profile.strengths) > 0
        
        # Only ask if we haven't received a valid response yet
        if not has_certifications:
            return "Have you completed any certifications or courses outside of your regular studies? Please list them if you have any."
        elif not has_extracurriculars:
            return "Are you involved in any extracurricular activities or clubs? If yes, please tell me about them."
        elif not has_careers:
            return "Do you have any particular careers in mind that interest you? What kind of work would you like to do?"
        elif not has_strengths:
            return "What would you say are your main strengths or skills? This helps me better understand your potential."
        
        # If we have all the information, move to recommendations
        return "I think I have a good understanding of your profile now. Would you like to see my recommendations?"
    
    async def _generate_profile_summary(self, profile: StudentProfile) -> str:
        """Generate summary of collected profile information"""
        subjects_str = "\n".join([
            f"- {s.name} ({s.grade or 'No grade'}, {'Favorite' if s.is_favorite else 'Not favorite'})"
            for s in profile.subjects
        ])
        
        summary = f"""Great! Here's what I know about you:
        
        Name: {profile.name}
        Academic Level: {profile.academic_level}
        
        Subjects:
        {subjects_str}
        
        Interests: {', '.join(profile.interests)}
        """
        
        if profile.certifications:
            summary += f"\nCertifications: {', '.join(profile.certifications)}"
        if profile.extracurriculars:
            summary += f"\nExtracurricular Activities: {', '.join(profile.extracurriculars)}"
        if profile.career_inclinations:
            summary += f"\nCareer Interests: {', '.join(profile.career_inclinations)}"
        if profile.strengths:
            summary += f"\nStrengths: {', '.join(profile.strengths)}"
        
        summary += "\n\nIs this information correct? Say 'yes' to proceed with recommendations, or let me know what needs to be updated."
        
        return summary
    
    async def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations for user presentation"""
        if not recommendations:
            return "I apologize, but I couldn't generate any recommendations at this time. Please try again."
        
        response = "Based on your profile, I recommend these university specializations:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec['specialization']}\n"
            response += f"   Reasoning: {rec['reasoning']}\n"
            response += f"   Key Subjects: {', '.join(rec['key_subjects'])}\n"
            response += f"   Potential Careers: {', '.join(rec['career_prospects'])}\n\n"
        
        response += "Which specialization would you like to explore further? (Respond with the number or name)"
        
        return response
    
    async def _process_career_path_request(self, state: ConversationState, message: str) -> str:
        """Process career path exploration request"""
        try:
            # Try to parse which specialization the user wants to explore
            selection = int(message.strip()[0]) - 1
            specialization = state.recommendations[selection]["specialization"]
        except:
            # If parsing fails, try to match specialization by name
            for rec in state.recommendations:
                if rec["specialization"].lower() in message.lower():
                    specialization = rec["specialization"]
                    break
            else:
                return "I'm not sure which specialization you're interested in. Could you specify by number (1, 2, etc.) or name?"
        
        # Get career paths for selected specialization
        career_paths = await self.career_path_engine.get_career_paths(
            specialization,
            state.profile.dict()
        )
        
        # Format response
        response = f"Here are some promising career paths for {specialization}:\n\n"
        
        for path in career_paths:
            response += f"• {path['career_path']}\n"
            response += f"  Description: {path['description']}\n"
            response += f"  Required Skills: {', '.join(path['required_skills'])}\n"
            response += f"  Career Progression: {path['progression']}\n"
            response += f"  Required Education: {', '.join(path['education'])}\n\n"
        
        response += "Would you like to explore another specialization or get more details about any of these career paths?"
        
        return response 