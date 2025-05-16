import time
from collections import defaultdict
from datetime import datetime, timedelta
from config import get_settings

settings = get_settings()

class RateLimiter:
    def __init__(self):
        self._requests = defaultdict(list)
        self._cleanup_interval = 60  # Cleanup old entries every minute
        self._last_cleanup = time.time()
    
    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self._last_cleanup > self._cleanup_interval:
            await self._cleanup()
            self._last_cleanup = current_time
        
        # Get request history for this session
        requests = self._requests[session_id]
        
        # Remove requests older than 1 minute
        minute_ago = current_time - 60
        requests = [req for req in requests if req > minute_ago]
        
        # Check if within limit
        if len(requests) >= settings.RATE_LIMIT_PER_MINUTE:
            raise Exception("Rate limit exceeded. Please wait before sending more requests.")
        
        # Add current request
        requests.append(current_time)
        self._requests[session_id] = requests
        
        return True
    
    async def _cleanup(self):
        """Remove old entries from request history"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        for session_id in list(self._requests.keys()):
            # Keep only requests from last minute
            self._requests[session_id] = [
                req for req in self._requests[session_id]
                if req > minute_ago
            ]
            
            # Remove empty session entries
            if not self._requests[session_id]:
                del self._requests[session_id] 