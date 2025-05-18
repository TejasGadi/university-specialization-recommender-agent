import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json
from ..models.student_profile import StudentProfile
from ..utils.rate_limiter import RateLimiter
from ..engines.fallback_recommendation_engine import FallbackRecommendationEngine
from ..engines.recommendation_engine import RecommendationEngine
from ..engines.career_path_engine import CareerPathEngine
from ..extractors.profile_extractor import ProfileExtractor
from config import get_settings
import logging

logger = logging.getLogger(__name__)
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
    intent: Optional[str] = None  # Store detected intent for context

class ConversationManager:
    def __init__(self):
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

            return response

        except asyncio.TimeoutError:
            error_msg = "I need a moment to process your request. Could you please repeat or simplify your question?"
            state.chat_history.append(Message(role="assistant", content=error_msg))
            return error_msg
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            error_msg = f"I apologize, but I encountered an error. Please try again."
            state.chat_history.append(Message(role="assistant", content=error_msg))
            return error_msg

    async def _process_message_internal(self, state: ConversationState, message: str) -> str:
        """Enhanced internal message processing with dynamic flow"""
        
        # First-time welcome message
        if state.stage == "welcome":
            state.stage = "profile_collection"
            welcome_msg = """Welcome! I'm your university specialization advisor. I'll help you find the best academic path 
                          based on your background and interests. Let's start by getting to know you. 
                          What subjects have you studied so far, and which ones do you enjoy most?"""
            return welcome_msg
            
        # For all other states, we'll first detect the user's intent to allow dynamic flow
        intent, confidence = await self._detect_intent(state, message)
        logger.info(f"Detected intent: {intent} with confidence: {confidence}")
        
        # Store intent for context in future interactions
        state.intent = intent
        
        # Dynamic stage transitions based on intent
        if intent == "update_profile" and confidence > 0.7:
            # User wants to update profile info from any stage
            if state.stage != "profile_collection":
                state.stage = "profile_collection"
                # Update profile with new information
                state.profile = await self._update_profile(state.profile, message)
                return await self._generate_dynamic_question(state)
            
        elif intent == "request_recommendations" and confidence > 0.7:
            # User explicitly wants recommendations from any stage
            if state.profile and state.profile.completion_percentage() >= 70:
                state.stage = "recommendation"
                # Generate recommendations immediately if profile is complete enough
                recommendations = await self._generate_recommendations_with_context(state)
                state.recommendations = recommendations
                return await self._format_recommendations(recommendations)
            else:
                # Not enough profile info to generate recommendations
                return await self._generate_dynamic_question(state)
                
        elif intent == "explore_career_paths" and confidence > 0.7 and state.recommendations:
            # User wants to explore career paths for a specialization
            state.stage = "career_paths"
            # Try to extract which specialization they're interested in
            specialization = await self._extract_specialization_mention(message, state.recommendations)
            if specialization:
                return await self._process_career_path_request(state, specialization)
            else:
                return "Which specialization would you like to explore career paths for?"
        
        # Process based on current stage with dynamic intelligence
        if state.stage == "profile_collection":
            # Update profile with new information
            updated_profile = await self._update_profile(state.profile, message)
            state.profile = updated_profile
            
            # Check if profile is complete enough for recommendations
            if updated_profile.completion_percentage() >= 95:
                missing_fields = updated_profile.get_missing_fields()
                if not missing_fields:
                    # Generate recommendations immediately if profile is complete
                    recommendations = await self._generate_recommendations_with_context(state)
                    state.recommendations = recommendations
                    return await self._format_recommendations(recommendations)
            
            # Generate dynamic question to gather more information
            return await self._generate_dynamic_question(state)
            
        elif state.stage == "recommendation":
            # Check if user confirms profile summary
            if "yes" in message.lower() or "correct" in message.lower() or intent == "confirm":
                # Generate recommendations with chat context
                recommendations = await self._generate_recommendations_with_context(state)
                state.recommendations = recommendations
                return await self._format_recommendations(recommendations)
            elif intent == "reject" or "no" in message.lower() or "incorrect" in message.lower():
                # User indicates profile summary is incorrect
                return "Let's update your profile information. What would you like to change?"
            else:
                # Process as additional profile information
                updated_profile = await self._update_profile(state.profile, message)
                state.profile = updated_profile
                # Generate recommendations with updated profile
                recommendations = await self._generate_recommendations_with_context(state)
                state.recommendations = recommendations
                return await self._format_recommendations(recommendations)
                
        elif state.stage == "career_paths":
            # Check if user is asking about a different specialization
            if intent == "change_specialization" or intent == "explore_different_specialization":
                return await self._process_career_path_request(state, message)
            elif intent == "request_more_recommendations":
                # User wants more specialization recommendations
                state.stage = "recommendation"
                recommendations = await self._generate_recommendations_with_context(state)
                state.recommendations = recommendations
                return await self._format_recommendations(recommendations)
            else:
                # Continue exploring career paths
                return await self._process_career_path_request(state, message)
        
        # Fallback - use dynamic response generation
        return await self._generate_dynamic_response(state, message)

    async def _detect_intent(self, state: ConversationState, message: str) -> Tuple[str, float]:
        """Detect user intent to enable dynamic conversation flow"""
        
        # Create system prompt for intent detection
        system_prompt = """You are an AI assistant analyzing user intent in a conversation with a university specialization advisor.
        Analyze the message and determine the primary intent. Respond with a JSON object containing:
        1. "intent": One of the following intents:
           - "update_profile" (user is sharing personal information or updating profile)
           - "request_recommendations" (user wants specialization recommendations)
           - "explore_career_paths" (user wants to explore career paths for a specialization)
           - "confirm" (user is confirming or agreeing with something)
           - "reject" (user is rejecting or disagreeing with something)
           - "change_specialization" (user wants to explore a different specialization)
           - "request_more_recommendations" (user wants additional specialization options)
           - "general_question" (user is asking a general question)
        2. "confidence": A float value between 0.0 and 1.0 indicating confidence in the intent detection
        
        Current conversation stage: {stage}
        User profile completion: {profile_completion}%
        """
        
        profile_completion = state.profile.completion_percentage() if state.profile else 0
        prompt = system_prompt.format(stage=state.stage, profile_completion=profile_completion)
        
        try:
            # Get recent conversation history for context
            messages = [SystemMessage(content=prompt)]
            for msg in state.chat_history[-5:]:  # Last 5 messages
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Get intent detection from chat model
            response = await self.chat_model.ainvoke(messages)
            
            # Parse response as JSON
            try:
                intent_data = json.loads(response.content)
                return intent_data.get("intent", "general_question"), intent_data.get("confidence", 0.5)
            except json.JSONDecodeError:
                logger.error("Failed to parse intent JSON")
                return "general_question", 0.5
                
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return "general_question", 0.5

    async def _extract_specialization_mention(self, message: str, recommendations: list) -> Optional[str]:
        """Extract which specialization the user is interested in exploring"""
        try:
            # Try to parse by number
            for i in range(1, len(recommendations) + 1):
                if str(i) in message:
                    return recommendations[i-1]["specialization"]
            
            # Try to match by name
            for rec in recommendations:
                if rec["specialization"].lower() in message.lower():
                    return rec["specialization"]
            
            # Use LLM to find the specialization if direct matching fails
            system_prompt = """The user is responding to a list of specialization recommendations. 
            Determine which specialization they are referring to in their message. 
            Available specializations:
            {}
            
            Respond with ONLY the exact name of the specialization they're referring to, or "unknown" if you can't determine it."""
            
            spec_list = "\n".join([f"{i+1}. {rec['specialization']}" for i, rec in enumerate(recommendations)])
            prompt = system_prompt.format(spec_list)
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=message)
            ]
            
            response = await self.chat_model.ainvoke(messages)
            extracted = response.content.strip()
            
            # Check if the extraction matches any recommendation
            for rec in recommendations:
                if rec["specialization"].lower() == extracted.lower():
                    return rec["specialization"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting specialization: {str(e)}")
            return None

    async def _update_profile(self, profile: Optional[StudentProfile], message: str) -> StudentProfile:
        """Update profile with information extracted from message"""
        # Extract new information from message
        extracted_info = await self.profile_extractor.extract_profile_info(message, profile)
        logger.info(f"Extracted info: {extracted_info}")

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
                logger.error(f"Error creating profile: {str(e)}")
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
                
                # Only update fields with new information
                for field, value in extracted_info.items():
                    if value:  # Only update if value is not empty
                        if field in ["subjects", "interests", "certifications", "extracurriculars", "career_inclinations", "strengths"]:
                            # For list fields, merge existing with new values
                            current_values = set(updated_info.get(field, []))
                            new_values = set(value)
                            updated_info[field] = list(current_values.union(new_values))
                        else:
                            # For scalar fields, replace with new value
                            updated_info[field] = value

                # Create new profile with updated information
                return StudentProfile(**updated_info)

            except Exception as e:
                logger.error(f"Error updating profile: {str(e)}")
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
1. Ask ONE natural follow-up question to gather missing information
2. Reference previous answers in your question
3. Make connections between shared interests and potential academic paths
4. Keep responses conversational but focused on gathering profile information
5. If all essential information is gathered, ask if they want to see recommendations
6. Keep your response brief and conversational - one question at a time

Current conversation stage: {state.stage}
Last detected intent: {state.intent or 'None'}

Remember, you should prioritize gathering information about subjects they enjoy and perform well in, their interests, and career goals."""

        messages.insert(0, SystemMessage(content=system_prompt))

        try:
            response = await self.chat_model.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating dynamic question: {str(e)}")
            # Fallback to static question generation
            return await self._generate_next_question(state.profile)

    async def _generate_next_question(self, profile: StudentProfile) -> str:
        """Fallback method to generate next question based on missing profile information"""
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

        response += "Which specialization would you like to explore further? (Respond with the number or name, or ask me about career paths for any of these options)"

        return response

    async def _process_career_path_request(self, state: ConversationState, message: str) -> str:
        """Process career path exploration request - enhanced to handle both string message and direct specialization"""
        try:
            # If message is a direct specialization name
            if isinstance(message, str) and any(rec["specialization"] == message for rec in state.recommendations):
                specialization = message
            else:
                # Try to parse which specialization the user wants to explore from a message
                try:
                    # First try by number
                    selection = int(message.strip()[0]) - 1
                    specialization = state.recommendations[selection]["specialization"]
                except:
                    # If parsing fails, try to match specialization by name
                    specialization = None
                    for rec in state.recommendations:
                        if rec["specialization"].lower() in message.lower():
                            specialization = rec["specialization"]
                            break
                    
                    if not specialization:
                        # Use LLM to extract specialization
                        extracted = await self._extract_specialization_mention(message, state.recommendations)
                        if extracted:
                            specialization = extracted
                        else:
                            return "I'm not sure which specialization you're interested in. Could you specify by number (1, 2, etc.) or name?"

            # Update selected specialization in state
            state.selected_specialization = specialization
            
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

            response += "Would you like to explore another specialization, get more details about any of these career paths, or update your profile information?"

            return response
            
        except Exception as e:
            logger.error(f"Error processing career path request: {str(e)}")
            return "I apologize, but I had trouble processing your request for career paths. Could you please specify which specialization you're interested in?"

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
            generate 3-4 university specialization recommendations in JSON format.
            
            Student Profile:
            {state.profile.dict()}
            
            Instructions:
            1. Consider the entire conversation context
            2. Focus on subjects they perform well in and enjoy
            3. Account for their stated interests and career goals
            4. Consider their academic level and any mentioned challenges
            5. Make connections with their extracurricular activities
            6. Be specific with specializations - provide concrete degree programs, not general fields
            
            Respond with a JSON array in this format:
            [
                {{
                    "specialization": "Name of Specialization",
                    "reasoning": "Detailed reasoning based on profile and conversation",
                    "key_subjects": ["Subject1", "Subject2", "Subject3"],
                    "career_prospects": ["Career1", "Career2", "Career3"]
                }}
            ]"""
            
            messages.insert(0, SystemMessage(content=system_prompt))

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
            return await self.fallback_recommendation_engine.generate_recommendations(state.profile)

    async def _generate_dynamic_response(self, state: ConversationState, message: str) -> str:
        """Generate dynamic conversational response when we can't categorize the message"""
        try:
            # Convert chat history to LangChain message format
            messages = []
            for msg in state.chat_history[-7:]:  # Use last 7 messages for context
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

            # Create system prompt for dynamic response
            system_prompt = f"""You are a university specialization advisor having a conversation with a student.
            
            Current Profile State:
            - Name: {state.profile.name if state.profile else 'Not provided'}
            - Academic Level: {state.profile.academic_level if state.profile else 'Not provided'}
            - Subjects: {', '.join(s.name for s in state.profile.subjects) if state.profile and state.profile.subjects else 'Not provided'}
            - Interests: {', '.join(state.profile.interests) if state.profile and state.profile.interests else 'Not provided'}
            
            Current conversation stage: {state.stage}
            Last detected intent: {state.intent or 'None'}
            
            Instructions:
            1. Respond to the user's message in a conversational and helpful manner
            2. If appropriate, guide the conversation back toward their academic specialization needs
            3. Keep responses concise but informative
            4. Always aim to be helpful while moving the conversation forward
            5. If user seems to be done with one topic, guide them toward the next logical step in the advising process"""

            messages.insert(0, SystemMessage(content=system_prompt))

            # Get response from chat model
            response = await self.chat_model.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating dynamic response: {str(e)}")
            return "I'm here to help with your academic specialization questions. Could you tell me more about your academic interests or what you'd like to explore next?"