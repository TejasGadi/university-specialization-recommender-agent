from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from typing import Dict

from src.managers.conversation_manager import ConversationManager
from config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
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

@app.get("/")
async def root():
    """Redirect to static/index.html"""
    return {"message": "Please visit /static/index.html to use the chat interface"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    connections[session_id] = websocket
    
    try:
        # Send welcome message
        await websocket.send_text(
            json.dumps({
                "type": "message",
                "content": "Welcome! I'm your university specialization advisor. How can I help you today?"
            })
        )
        
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            try:
                # Process message
                response = await conversation_manager.process_message(session_id, message)
                
                # Send response
                await websocket.send_text(
                    json.dumps({
                        "type": "message",
                        "content": response
                    })
                )
                
            except Exception as e:
                # Send error message
                await websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "content": str(e)
                    })
                )
    
    except WebSocketDisconnect:
        # Clean up on disconnect
        connections.pop(session_id, None)
    
    except Exception as e:
        # Handle other errors
        if session_id in connections:
            await websocket.send_text(
                json.dumps({
                    "type": "error",
                    "content": f"An error occurred: {str(e)}"
                })
            )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 