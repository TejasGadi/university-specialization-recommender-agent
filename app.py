from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from typing import Dict, Optional
import asyncio
import logging
from datetime import datetime

from src.managers.conversation_manager import ConversationManager
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use settings instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active websocket connections
connections: Dict[str, WebSocket] = {}

# Initialize conversation manager
conversation_manager = ConversationManager()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_activity: Dict[str, datetime] = {}
        self.cleanup_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, session_id: str):
        try:
            await websocket.accept()
            self.active_connections[session_id] = websocket
            self.last_activity[session_id] = datetime.now()
            
            # Start cleanup task if not running
            if not self.cleanup_task or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        except Exception as e:
            logger.error(f"Error connecting websocket for session {session_id}: {str(e)}")
            raise

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.last_activity.pop(session_id, None)

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {str(e)}")
                await self.disconnect(session_id)

    async def _cleanup_inactive_sessions(self):
        """Cleanup inactive sessions periodically"""
        while True:
            try:
                current_time = datetime.now()
                inactive_sessions = [
                    session_id for session_id, last_active in self.last_activity.items()
                    if (current_time - last_active).total_seconds() > settings.SESSION_TIMEOUT
                ]
                
                for session_id in inactive_sessions:
                    logger.info(f"Cleaning up inactive session: {session_id}")
                    self.disconnect(session_id)
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)

connection_manager = ConnectionManager()

@app.get("/")
async def root():
    """Redirect to static/index.html"""
    return {"message": "Please visit /static/index.html to use the chat interface"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(connection_manager.active_connections)
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await connection_manager.connect(websocket, session_id)
        
        # Send welcome message
        await connection_manager.send_message(
            session_id,
            {
                "type": "message",
                "content": "Welcome! I'm your university specialization advisor. How can I help you today?"
            }
        )
        
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                
                # Update last activity
                connection_manager.last_activity[session_id] = datetime.now()
                
                # Send typing indicator
                await connection_manager.send_message(
                    session_id,
                    {"type": "status", "content": "typing"}
                )
                
                try:
                    # Process message
                    response = await conversation_manager.process_message(session_id, message)
                    
                    # Send response
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "message",
                            "content": response
                        }
                    )
                    
                except asyncio.TimeoutError:
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "error",
                            "content": "Response took too long. Please try again with a simpler question."
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "error",
                            "content": "An error occurred while processing your message. Please try again."
                        }
                    )
                
                finally:
                    # Clear typing indicator
                    await connection_manager.send_message(
                        session_id,
                        {"type": "status", "content": "idle"}
                    )
            except Exception as e:
                logger.error(f"Error handling message: {str(e)}")
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "error",
                        "content": "An error occurred while handling your message. Please try again."
                    }
                )
    
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
        logger.info(f"Client disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        connection_manager.disconnect(session_id)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )