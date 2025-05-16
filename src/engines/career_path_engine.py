from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import json
import os
import logging
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class CareerPath(BaseModel):
    """Pydantic model for career path recommendations"""
    career_path: str = Field(description="Name of the career path")
    description: str = Field(description="Detailed description of the career path")
    required_skills: List[str] = Field(description="List of required skills for this career path")
    progression: str = Field(description="Career progression path from entry level to senior positions")
    education: List[str] = Field(description="Required education and certifications")

class CareerPathResponse(BaseModel):
    """Pydantic model for career path response"""
    recommendations: List[CareerPath] = Field(description="List of career path recommendations")

# Sample career paths data structure
SAMPLE_CAREER_PATHS = [
    {
        "title": "Software Engineer",
        "description": "Develops software applications and systems",
        "specializations": ["Computer Science", "Software Engineering"],
        "required_skills": ["Programming", "Problem Solving", "Software Design"],
        "industries": ["Technology", "Finance", "Healthcare"],
        "progression": "Junior Developer → Senior Developer → Tech Lead → Software Architect",
        "education": ["Bachelor's in Computer Science", "Software Engineering Certifications"]
    },
    {
        "title": "Data Scientist",
        "description": "Analyzes complex data to help organizations make better decisions",
        "specializations": ["Data Science", "Computer Science", "Statistics"],
        "required_skills": ["Machine Learning", "Statistics", "Programming"],
        "industries": ["Technology", "Finance", "Healthcare", "Research"],
        "progression": "Junior Data Scientist → Senior Data Scientist → Lead Data Scientist → Chief Data Scientist",
        "education": ["Master's in Data Science", "Statistics", "Machine Learning"]
    }
]

class CareerPathEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.vector_store = None
        self.qa_chain = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load vector store"""
        vector_store_path = settings.VECTOR_STORE_PATH
        
        # Create vector store directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Check if vector store exists
        if not os.path.exists(os.path.join(vector_store_path, "chroma.sqlite3")):
            # Create new vector store with sample data
            self._create_vector_store()
        else:
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embeddings
            )
        
        # Initialize LLM with structured output
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Create LLM with structured output
        self.structured_llm = llm.with_structured_output(CareerPathResponse)
        
        # Create a custom prompt template
        self.prompt_template = PromptTemplate(
            template="""Based on the following context and query, provide career path recommendations.
            
            Context: {context}
            
            Query: {query}
            
            Analyze the profile and context to provide detailed career path recommendations. Include:
            1. Specific career paths that match their interests and skills
            2. Required skills and education for each path
            3. Typical career progression
            4. Detailed description of each role
            
            Ensure recommendations are specific and actionable.""",
            input_variables=["context", "query"]
        )
        
        # Create the chain using modern patterns
        retriever = self.vector_store.as_retriever()
        
        # Define the context processing function
        def get_context(query_dict: dict) -> dict:
            docs = retriever.invoke(query_dict["query"])
            return {
                "context": "\n".join(doc.page_content for doc in docs),
                "query": query_dict["query"]
            }
        
        # Setup the chain
        self.qa_chain = (
            RunnablePassthrough() 
            | get_context
            | self.prompt_template
            | self.structured_llm
        )
    
    def _create_vector_store(self):
        """Create new vector store with sample career paths data"""
        try:
            # Save sample data to temporary file
            with open("temp_career_paths.json", "w") as f:
                json.dump(SAMPLE_CAREER_PATHS, f)
            
            # Load and split documents
            loader = JSONLoader(
                file_path="temp_career_paths.json",
                jq_schema=".",
                text_content=False
            )
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create and persist vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=settings.VECTOR_STORE_PATH
            )
            self.vector_store.persist()
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if os.path.exists("temp_career_paths.json"):
                os.remove("temp_career_paths.json")
    
    async def get_career_paths(self, specialization: str, profile: dict) -> List[Dict]:
        """Get relevant career paths for given specialization and profile"""
        # Construct query based on specialization and profile
        query = f"""
        What are the most suitable career paths for someone interested in {specialization}
        with the following profile:
        - Interests: {', '.join(profile.get('interests', []))}
        - Strengths: {', '.join(profile.get('strengths', []))}
        - Academic Background: {profile.get('academic_level', 'Not specified')}
        """
        
        try:
            # Get response using the modern chain with structured output
            result = await self.qa_chain.ainvoke({"query": query})
            
            # Convert Pydantic model to dict for compatibility
            return [career_path.model_dump() for career_path in result.recommendations]
            
        except Exception as e:
            logger.error(f"Error generating career paths: {str(e)}")
            return [{
                "career_path": f"{specialization} Professional",
                "description": f"Career path in {specialization}. Unable to generate specific details due to processing issues.",
                "required_skills": ["Core skills in " + specialization, "Problem Solving", "Communication"],
                "progression": "Entry Level → Mid Level → Senior Level → Lead/Manager",
                "education": [f"Degree in {specialization}", "Relevant certifications"]
            }]
    
    async def add_career_path(self, career_path: Dict) -> bool:
        """Add new career path to the vector store"""
        try:
            # Convert career path to document format
            text = f"""
            Title: {career_path['title']}
            Description: {career_path['description']}
            Specializations: {', '.join(career_path['specializations'])}
            Required Skills: {', '.join(career_path['required_skills'])}
            Industries: {', '.join(career_path['industries'])}
            Progression: {career_path['progression']}
            Education: {', '.join(career_path['education'])}
            """
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=[text],
                metadatas=[career_path]
            )
            
            # Persist changes
            self.vector_store.persist()
            logger.info(f"Successfully added career path: {career_path['title']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding career path: {str(e)}")
            return False 