import asyncio
from typing import Optional, Dict, Any, List, Tuple, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from ..models.student_profile import StudentProfile, Subject
from ..utils.rate_limiter import RateLimiter
from ..engines.fallback_recommendation_engine import FallbackRecommendationEngine
from ..engines.recommendation_engine import RecommendationEngine
from ..engines.career_path_engine import CareerPathEngine
from ..extractors.profile_extractor import ProfileExtractor
from config import get_settings
import logging
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)
settings = get_settings()

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class IntentDetectionResult(BaseModel):
    intent: Literal[
        "update_profile",
        "request_recommendations",
        "explore_career_paths",
        "confirm",
        "reject",
        "change_specialization",
        "request_more_recommendations",
        "general_question"
    ] = Field(...)
    
    confidence: float = Field(..., ge=0.0, le=1.0)

class Specialization(BaseModel):
    """Pydantic model for specialization recommendation"""
    specialization: str = Field(..., description="Name of the specialization")
    reasoning: str = Field(..., description="Detailed reasoning based on profile and conversation")
    key_subjects: List[str] = Field(..., description="List of key subjects for this specialization")
    career_prospects: List[str] = Field(..., description="List of potential career paths")

class RecommendationList(BaseModel):
    """Pydantic model for list of specialization recommendations"""
    recommendations: List[Specialization] = Field(..., min_items=1, max_items=5)

class CareerPath(BaseModel):
    """Pydantic model for career path information"""
    career_path: str = Field(..., description="Name of the career path")
    description: str = Field(..., description="Detailed description of the career path")
    required_skills: List[str] = Field(..., description="List of required skills")
    progression: str = Field(..., description="Career progression path")
    education: List[str] = Field(..., description="Required education and qualifications")

class CareerPathList(BaseModel):
    """Pydantic model for list of career paths"""
    career_paths: List[CareerPath] = Field(..., min_items=1, max_items=5)

class SpecializationExtraction(BaseModel):
    """Pydantic model for specialization extraction result"""
    specialization: Optional[str] = Field(None, description="Extracted specialization name or None if not found")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the extraction")

class ConversationState(BaseModel):
    stage: str = "welcome"  # welcome, profile_collection, recommendation, career_paths
    profile: Optional[StudentProfile] = None
    last_interaction: datetime = datetime.now()
    recommendations: Optional[List[Specialization]] = None
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
        
        # Initialize intent detection chain
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentDetectionResult)
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are an AI assistant analyzing user intent in a conversation with a university specialization advisor.
Your task is to determine the user's primary intent from their message.

Current conversation stage: {stage}
Profile completion: {profile_completion}%

Available intents:
1. update_profile - User is providing or updating their profile information
2. request_recommendations - User wants to see specialization recommendations
3. explore_career_paths - User wants to explore career paths for a specialization
4. confirm - User is confirming or agreeing with something
5. reject - User is rejecting or disagreeing with something
6. change_specialization - User wants to explore a different specialization
7. request_more_recommendations - User wants more specialization options
8. general_question - Default for general queries or unclear intent

Analyze the message and determine the primary intent. Consider:
- The current conversation stage
- The message content and context
- Any explicit or implicit requests
- The natural flow of academic advising

{format_instructions}'''),
            ("human", "{message}")
        ])
        
        # Create the QA chain for intent detection
        self.intent_chain = (
            RunnablePassthrough() 
            | self._get_intent_context 
            | self.intent_prompt 
            | self.chat_model 
            | self.intent_parser
        )

    def _get_intent_context(self, input_dict: Dict) -> Dict:
        """Prepare context for intent detection"""
        state = input_dict["state"]
        message = input_dict["message"]
        
        # Get profile completion
        profile_completion = state.profile.completion_percentage() if state.profile else 0
        
        return {
            "stage": state.stage,
            "profile_completion": profile_completion,
            "message": message,
            "format_instructions": self.intent_parser.get_format_instructions()
        }

    async def _detect_intent(self, state: ConversationState, message: str) -> Tuple[str, float]:
        """Detect user intent using a QA chain"""
        try:
            # Prepare input for the chain
            chain_input = {
                "state": state,
                "message": message
            }
            
            # Run the chain
            result = await self.intent_chain.ainvoke(chain_input)
            
            # Log the successful intent detection
            logger.info(f"Intent detection result: {result.dict()}")
            return result.intent, result.confidence
            
        except Exception as e:
            logger.error(f"Intent detection failed: {str(e)}", exc_info=True)
            # Determine a more appropriate fallback intent based on the message content
            fallback_intent = self._determine_fallback_intent(message, state)
            return fallback_intent, 0.5

    async def _stream_response(self, text: str, websocket) -> None:
        """Helper method to stream response to the websocket with improved error handling"""
        if not websocket:
            return
            
        try:
            # Split text into sentences and stream each one
            sentences = text.split('. ')
            current_text = ""
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                try:
                    current_text += sentence.strip()
                    
                    # Send the current chunk
                    await websocket.send_json({
                        "type": "stream",
                        "content": current_text
                    })
                    
                    # Add period and space after each sentence except the last one
                    if i < len(sentences) - 1:
                        current_text += ". "
                    
                    await asyncio.sleep(0.05)  # Reduced delay between sentences
                    
                except Exception as e:
                    logger.error(f"Error sending stream chunk: {str(e)}")
                    raise  # Re-raise to handle in outer try-catch
            
            # Send the final complete message
            await websocket.send_json({
                "type": "end_stream",
                "content": text.strip()
            })
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
            # Don't try to send error message - let the caller handle it
            raise

    async def process_message(self, session_id: str, message: str, websocket=None) -> str:
        """Process incoming message and return response"""
        try:
            logger.info(f"Processing message for session {session_id}: {message}")
            
            # Check rate limit
            await self.rate_limiter.check_rate_limit(session_id)

            # Get or create conversation state
            state = self.states.get(session_id, ConversationState())
            if not state:
                state = ConversationState()
                self.states[session_id] = state

            # Add user message to chat history
            state.chat_history.append(Message(role="user", content=message))

            try:
                logger.info(f"Starting message processing for session {session_id} in stage: {state.stage}")
                
                # Process message without timeout
                response = await self._process_message_internal(state, message, websocket)

                logger.info(f"Generated response for session {session_id}: {response[:100]}...")

                # Add assistant response to chat history
                state.chat_history.append(Message(role="assistant", content=response))

                # Update state
                state.last_interaction = datetime.now()
                self.states[session_id] = state

                return response

            except Exception as e:
                logger.error(f"Error in process_message: {str(e)}", exc_info=True)
                error_msg = (
                    "I encountered an issue processing your response. "
                    "Could you try again? I'll take as much time as needed to process it properly."
                )
                
                state.chat_history.append(Message(role="assistant", content=error_msg))
                return error_msg

        except Exception as e:
            logger.error(f"Critical error in process_message for session {session_id}: {str(e)}", exc_info=True)
            return "I apologize, but I encountered a system error. Please try again."

    async def _process_message_internal(self, state: ConversationState, message: str, websocket=None) -> str:
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
                # Print profile summary before generating recommendations
                profile_summary = await self._generate_profile_summary(state.profile)
                logger.info(f"Profile before recommendations:\n{profile_summary}")
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
                    # Print profile summary before generating recommendations
                    profile_summary = await self._generate_profile_summary(updated_profile)
                    logger.info(f"Profile before recommendations:\n{profile_summary}")
                    # Generate recommendations immediately if profile is complete
                    recommendations = await self._generate_recommendations_with_context(state)
                    state.recommendations = recommendations
                    return await self._format_recommendations(recommendations)
            
            # Generate dynamic question to gather more information
            return await self._generate_dynamic_question(state)
            
        elif state.stage == "recommendation":
            # Check if user confirms profile summary
            if "yes" in message.lower() or "correct" in message.lower() or intent == "confirm":
                # Print profile summary before generating recommendations
                profile_summary = await self._generate_profile_summary(state.profile)
                logger.info(f"Profile before recommendations:\n{profile_summary}")
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
                # Print profile summary before generating recommendations
                profile_summary = await self._generate_profile_summary(updated_profile)
                logger.info(f"Profile before recommendations:\n{profile_summary}")
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

    def _determine_fallback_intent(self, message: str, state: ConversationState) -> str:
        """Determine a fallback intent based on message content and conversation state"""
        message_lower = message.lower()
        
        # Check for common confirmations
        if any(word in message_lower for word in ['yes', 'yeah', 'correct', 'right', 'sure']):
            return "confirm"
            
        # Check for common rejections
        if any(word in message_lower for word in ['no', 'nope', 'incorrect', 'wrong']):
            return "reject"
            
        # Check if in recommendation stage and mentioning specializations
        if state.stage == "recommendation" and state.recommendations:
            for rec in state.recommendations:
                if rec.specialization.lower() in message_lower:
                    return "explore_career_paths"
                    
        # If in profile collection stage, likely providing profile info
        if state.stage == "profile_collection":
            return "update_profile"
            
        # Default fallback
        return "general_question"

    async def _extract_specialization_mention(self, message: str, recommendations: List[Specialization]) -> Optional[str]:
        """Extract which specialization the user is interested in exploring"""
        try:
            # Try to parse by number
            for i in range(1, len(recommendations) + 1):
                if str(i) in message:
                    return recommendations[i-1].specialization
            
            # Try to match by name
            for rec in recommendations:
                if rec.specialization.lower() in message.lower():
                    return rec.specialization
            
            # Use LLM to find the specialization if direct matching fails
            system_prompt = """The user is responding to a list of specialization recommendations. 
            Determine which specialization they are referring to in their message.
            
            Available specializations:
            {}
            
            Return a structured response indicating the specialization name and your confidence in the extraction."""
            
            spec_list = "\n".join([f"{i+1}. {rec.specialization}" for i, rec in enumerate(recommendations)])
            prompt = system_prompt.format(spec_list)
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=message)
            ]
            
            # Use structured output for specialization extraction
            structured_llm = self.chat_model.with_structured_output(SpecializationExtraction)
            result: SpecializationExtraction = await structured_llm.ainvoke(messages)
            
            if result.specialization and result.confidence > 0.7:
                # Verify the extracted specialization exists in recommendations
                for rec in recommendations:
                    if rec.specialization.lower() == result.specialization.lower():
                        return rec.specialization
            
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
                    subjects=[Subject(name="General Studies", is_favorite=True)],
                    interests=["General Studies"]
                )
        else:
            try:
                # Create a new dictionary with all existing profile data
                updated_info = profile.dict()
                
                # Only update fields with new information
                for field, value in extracted_info.items():
                    if value:  # Only update if value is not empty
                        if field == "subjects":
                            # For subjects, merge based on subject names to avoid duplicates
                            existing_subjects = {s["name"]: s for s in updated_info.get("subjects", [])}
                            for new_subject in value:
                                subject_name = new_subject["name"] if isinstance(new_subject, dict) else new_subject.name
                                if subject_name in existing_subjects:
                                    # Update existing subject if new one has more information
                                    existing_subject = existing_subjects[subject_name]
                                    if isinstance(new_subject, dict):
                                        if new_subject.get("grade"):
                                            existing_subject["grade"] = new_subject["grade"]
                                        if new_subject.get("is_favorite"):
                                            existing_subject["is_favorite"] = new_subject["is_favorite"]
                                else:
                                    # Add new subject
                                    if isinstance(new_subject, dict):
                                        existing_subjects[subject_name] = new_subject
                                    else:
                                        existing_subjects[subject_name] = new_subject.dict()
                            updated_info["subjects"] = list(existing_subjects.values())
                        elif field in ["interests", "certifications", "extracurriculars", "career_inclinations", "strengths", "challenges"]:
                            # For simple list fields, merge using sets
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

    async def _format_recommendations(self, recommendations: List[Specialization]) -> str:
        """Format recommendations for user presentation"""
        if not recommendations:
            return "I apologize, but I couldn't generate any recommendations at this time. Please try again."

        response = "Based on your profile, I recommend these university specializations:\n\n"

        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec.specialization}\n"
            response += f"   Reasoning: {rec.reasoning}\n"
            response += f"   Key Subjects: {', '.join(rec.key_subjects)}\n"
            response += f"   Potential Careers: {', '.join(rec.career_prospects)}\n\n"

        response += "Which specialization would you like to explore further? (Respond with the number or name, or ask me about career paths for any of these options)"

        return response

    async def _process_career_path_request(self, state: ConversationState, message: str) -> str:
        """Process career path exploration request - enhanced to handle both string message and direct specialization"""
        try:
            # If message is a direct specialization name
            if isinstance(message, str) and any(rec.specialization == message for rec in state.recommendations):
                specialization = message
            else:
                # Try to extract which specialization the user wants to explore
                specialization = await self._extract_specialization_mention(message, state.recommendations)
                if not specialization:
                    return "I'm not sure which specialization you're interested in. Could you specify by number (1, 2, etc.) or name?"

            # Update selected specialization in state
            state.selected_specialization = specialization
            
            # Create system prompt for career paths
            system_prompt = f"""You are a university specialization advisor providing career path information.
            Generate detailed career paths for the {specialization} specialization.
            Consider the student's profile and interests when describing the paths.
            
            Student Profile:
            {state.profile.dict()}
            
            Each career path must include:
            - A specific career path name
            - Detailed description of the role and responsibilities
            - Required skills and competencies
            - Typical career progression
            - Required education and qualifications"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"What are the career paths for {specialization}?")
            ]
            
            # Get career paths using structured output
            structured_llm = self.chat_model.with_structured_output(CareerPathList)
            result: CareerPathList = await structured_llm.ainvoke(messages)
            
            # Format response
            response = f"Here are some promising career paths for {specialization}:\n\n"

            for path in result.career_paths:
                response += f"• {path.career_path}\n"
                response += f"  Description: {path.description}\n"
                response += f"  Required Skills: {', '.join(path.required_skills)}\n"
                response += f"  Career Progression: {path.progression}\n"
                response += f"  Required Education: {', '.join(path.education)}\n\n"

            response += "Would you like to explore another specialization, get more details about any of these career paths, or update your profile information?"

            return response
            
        except Exception as e:
            logger.error(f"Error processing career path request: {str(e)}")
            return "I apologize, but I had trouble processing your request for career paths. Could you please specify which specialization you're interested in?"

    async def _generate_recommendations_with_context(self, state: ConversationState) -> List[Specialization]:
        """Generate recommendations using chat history context"""
        try:
            # Print detailed profile information before generating recommendations
            logger.info(f"Generating recommendations for profile:\n{state.profile.dict()}")
            
            # Calculate and log profile completion percentage
            completion = self.calculate_profile_completion(state)
            logger.info(f"Profile completion: {completion}%")
            
            # Log missing fields
            required_missing = self.get_missing_required_fields(state)
            optional_missing = self.get_missing_optional_fields(state)
            if required_missing:
                logger.info(f"Missing required fields: {required_missing}")
            if optional_missing:
                logger.info(f"Missing optional fields: {optional_missing}")

            # Prepare messages with chat history context
            messages = []
            for msg in state.chat_history[-10:]:  # Use last 10 messages for context
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

            # Create system prompt for recommendations
            system_prompt = f"""You are a university specialization advisor. Based on the student's profile and conversation history, 
            generate 3-4 university specialization recommendations.
            
            Student Profile:
            {state.profile.dict()}
            
            Instructions:
            1. Consider the entire conversation context
            2. Focus on subjects they perform well in and enjoy
            3. Account for their stated interests and career goals
            4. Consider their academic level and any mentioned challenges
            5. Make connections with their extracurricular activities
            6. Be specific with specializations - provide concrete degree programs, not general fields
            
            Each recommendation must include:
            - A specific specialization name
            - Detailed reasoning based on the student's profile
            - List of key subjects required for the specialization
            - List of potential career prospects"""
            
            messages.insert(0, SystemMessage(content=system_prompt))

            # Get recommendations using structured output
            structured_llm = self.chat_model.with_structured_output(RecommendationList)
            result: RecommendationList = await structured_llm.ainvoke(messages)
            
            return result.recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations with context: {str(e)}")
            # Fallback to regular recommendation engine
            fallback_recs = await self.fallback_recommendation_engine.generate_recommendations(state.profile)
            # Convert fallback recommendations to Specialization objects
            return [Specialization(**rec) for rec in fallback_recs]

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

    def calculate_profile_completion(self, state: ConversationState) -> float:
        """Calculate profile completion percentage based on filled fields"""
        if not state.profile:
            return 0.0
        
        # Required fields are weighted more heavily (70%)
        required_fields = {
            'name': bool(state.profile.name and state.profile.name not in ["Student", "Anonymous Student"]),
            'academic_level': bool(state.profile.academic_level),
            'subjects': bool(state.profile.subjects and len(state.profile.subjects) > 0 and not all(s.name == "General Studies" for s in state.profile.subjects)),
            'interests': bool(state.profile.interests and len(state.profile.interests) > 0 and "General Studies" not in state.profile.interests)
        }
        
        # Optional fields contribute 30%
        optional_fields = {
            'age': bool(state.profile.age),
            'certifications': bool(state.profile.certifications and len(state.profile.certifications) > 0),
            'extracurriculars': bool(state.profile.extracurriculars and len(state.profile.extracurriculars) > 0),
            'career_inclinations': bool(state.profile.career_inclinations and len(state.profile.career_inclinations) > 0),
            'strengths': bool(state.profile.strengths and len(state.profile.strengths) > 0),
            'challenges': bool(state.profile.challenges and len(state.profile.challenges) > 0)
        }
        
        # Calculate scores
        required_score = sum(required_fields.values()) * (70.0 / len(required_fields))
        optional_score = sum(optional_fields.values()) * (30.0 / len(optional_fields))
        
        return required_score + optional_score

    def get_missing_required_fields(self, state: ConversationState) -> List[str]:
        """Return list of missing required fields"""
        if not state.profile:
            return ["name", "academic_level", "subjects", "interests"]
        
        missing = []
        if not state.profile.name or state.profile.name in ["Student", "Anonymous Student"]:
            missing.append("name")
        if not state.profile.academic_level:
            missing.append("academic_level")
        if not state.profile.subjects or all(s.name == "General Studies" for s in state.profile.subjects):
            missing.append("subjects")
        if not state.profile.interests or "General Studies" in state.profile.interests:
            missing.append("interests")
        return missing

    def get_missing_optional_fields(self, state: ConversationState) -> List[str]:
        """Return list of missing optional fields"""
        if not state.profile:
            return ["age", "certifications", "extracurriculars", "career_inclinations", "strengths", "challenges"]
        
        missing = []
        if not state.profile.age:
            missing.append("age")
        if not state.profile.certifications:
            missing.append("certifications")
        if not state.profile.extracurriculars:
            missing.append("extracurriculars")
        if not state.profile.career_inclinations:
            missing.append("career_inclinations")
        if not state.profile.strengths:
            missing.append("strengths")
        if not state.profile.challenges:
            missing.append("challenges")
        return missing

    def is_profile_complete_enough(self, state: ConversationState) -> bool:
        """Check if profile is complete enough to proceed with recommendations"""
        completion = self.calculate_profile_completion(state)
        required_fields_missing = self.get_missing_required_fields(state)
        return completion >= 70.0 and not required_fields_missing