import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
import json

from ..models.student_profile import StudentProfile
from ..utils.cache import Cache
from ..utils.rate_limiter import RateLimiter
from ..engines.fallback_recommendation_engine import FallbackRecommendationEngine
from ..engines.recommendation_engine import RecommendationEngine
from ..engines.career_path_engine import CareerPathEngine
from ..extractors.profile_extractor import ProfileExtractor
from config import get_settings

settings = get_settings()

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ConversationState(BaseModel):
    stage: str = "welcome"  # welcome, profile_collection, recommendation, career_paths
    profile: Optional[StudentProfile] = None
    last_interaction: datetime = datetime.now()
    recommendations: Optional[list] = None
    selected_specialization: Optional[str] = None
    chat_history: List[Message] = []  # Add chat history

class ConversationManager:
    def __init__(self):
        self.cache = Cache()
        self.rate_limiter = RateLimiter()
        self.states: Dict[str, ConversationState] = {}
        self.profile_extractor = ProfileExtractor()
        self.fallback_recommendation_engine = FallbackRecommendationEngine()
        self.recommendation_engine = RecommendationEngine()
        self.career_path_engine = CareerPathEngine()
        self.chat_model = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
    
    async def process_message(self, session_id: str, message: str) -> str:
        """Process incoming message and return response within time constraint"""
        try:
            # Check rate limit
            await self.rate_limiter.check_rate_limit(session_id)
            
            # Get or create conversation state
            state = self.states.get(session_id, ConversationState())
            
            # Add user message to chat history
            state.chat_history.append(Message(role="user", content=message))
            
            # Try to get cached response
            cache_key = f"{session_id}:{message}"
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                # Add cached response to chat history
                state.chat_history.append(Message(role="assistant", content=cached_response))
                return cached_response
            
            # Process message with timeout
            response = await asyncio.wait_for(
                self._process_message_internal(state, message),
                timeout=settings.MAX_RESPONSE_TIME
            )
            
            # Add assistant response to chat history
            state.chat_history.append(Message(role="assistant", content=response))
            
            # Update state
            state.last_interaction = datetime.now()
            self.states[session_id] = state
            
            # Cache response
            await self.cache.set(cache_key, response)
            
            return response
            
        except asyncio.TimeoutError:
            error_msg = "I need a moment to process your request. Could you please repeat or simplify your question?"
            state.chat_history.append(Message(role="assistant", content=error_msg))
            return error_msg
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"
            state.chat_history.append(Message(role="assistant", content=error_msg))
            return error_msg
    
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
            if updated_profile.completion_percentage() >= 95:
                missing_fields = updated_profile.get_missing_fields()
                if not missing_fields:
                    state.stage = "recommendation"
                    return await self._generate_profile_summary(updated_profile)
            
            # Generate dynamic question
            return await self._generate_dynamic_question(state)
        
        elif state.stage == "recommendation":
            if "yes" in message.lower() or "correct" in message.lower():
                # Generate recommendations with chat context
                recommendations = await self._generate_recommendations_with_context(state)
                state.recommendations = recommendations
                state.stage = "career_paths"
                return await self._format_recommendations(recommendations)
            else:
                return await self._generate_dynamic_question(state)
        
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
    
    async def _generate_dynamic_question(self, state: ConversationState) -> str:
        """Generate dynamic questions based on conversation context and profile state"""
        # Convert chat history to LangChain message format
        messages = []
        for msg in state.chat_history[-5:]:  # Use last 5 messages for context
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
                
        # Create system prompt
        system_prompt = f"""You are a university specialization advisor having a conversation with a student.
Current Profile State:
- Name: {state.profile.name if state.profile else 'Not provided'}
- Academic Level: {state.profile.academic_level if state.profile else 'Not provided'}
- Subjects: {', '.join(s.name for s in state.profile.subjects) if state.profile and state.profile.subjects else 'Not provided'}
- Interests: {', '.join(state.profile.interests) if state.profile and state.profile.interests else 'Not provided'}
- Certifications: {', '.join(state.profile.certifications) if state.profile and state.profile.certifications else 'Not provided'}
- Extracurriculars: {', '.join(state.profile.extracurriculars) if state.profile and state.profile.extracurriculars else 'Not provided'}
- Career Inclinations: {', '.join(state.profile.career_inclinations) if state.profile and state.profile.career_inclinations else 'Not provided'}
- Strengths: {', '.join(state.profile.strengths) if state.profile and state.profile.strengths else 'Not provided'}

Missing Information: {', '.join(state.profile.get_missing_fields()) if state.profile else 'All fields'}

Instructions:
1. Ask natural follow-up questions to gather missing information
2. Reference previous answers in your questions
3. Make connections between shared interests and potential academic paths
4. Keep responses conversational but focused on gathering profile information
5. If all essential information is gathered, ask if they want to see recommendations

Current conversation stage: {state.stage}"""

        messages.insert(0, HumanMessage(content=system_prompt))
        
        try:
            response = await self.chat_model.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating dynamic question: {str(e)}")
            # Fallback to static question generation
            return await self._generate_next_question(state.profile)
    
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

    async def _generate_recommendations_with_context(self, state: ConversationState) -> List[Dict]:
        """Generate recommendations using chat history context"""
        try:
            # Prepare messages with chat history context
            messages = []
            for msg in state.chat_history[-10:]:  # Use last 10 messages for context
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
            
            # Create system prompt for recommendations
            system_prompt = f"""You are a university specialization advisor. Based on the student's profile and conversation history, 
            generate 2-3 university specialization recommendations in JSON format.

            Student Profile:
            {state.profile.dict()}

            Instructions:
            1. Consider the entire conversation context
            2. Focus on subjects they perform well in and enjoy
            3. Account for their stated interests and career goals
            4. Consider their academic level and any mentioned challenges
            5. Make connections with their extracurricular activities

            Respond with a JSON array in this format:
            [
                {{
                    "specialization": "Name of Specialization",
                    "reasoning": "Detailed reasoning based on profile and conversation",
                    "key_subjects": ["Subject1", "Subject2", "Subject3"],
                    "career_prospects": ["Career1", "Career2", "Career3"]
                }}
            ]"""

            messages.insert(0, HumanMessage(content=system_prompt))
            
            # Get recommendation from chat model
            response = await self.chat_model.ainvoke(messages)
            
            # Parse JSON response
            try:
                recommendations = json.loads(response.content)
                if isinstance(recommendations, list):
                    return recommendations
            except json.JSONDecodeError:
                logger.error("Failed to parse recommendations JSON")
            
            # Fallback to regular recommendation engine
            return await self.recommendation_engine.generate_recommendations(state.profile)
            
        except Exception as e:
            logger.error(f"Error generating recommendations with context: {str(e)}")
            return await self.recommendation_engine.generate_recommendations(state.profile)