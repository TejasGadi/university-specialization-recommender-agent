import redis
import json
from typing import Optional, Any
from config import get_settings

settings = get_settings()

class Cache:
    def __init__(self):
        self._redis = None
        if settings.REDIS_URL:
            self._redis = redis.from_url(settings.REDIS_URL)
        self._local_cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first if available
        if self._redis:
            try:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
            except:
                pass
        
        # Fallback to local cache
        return self._local_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        if ttl is None:
            ttl = settings.CACHE_TTL
            
        # Try Redis first if available
        if self._redis:
            try:
                self._redis.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )
                return True
            except:
                pass
        
        # Fallback to local cache
        self._local_cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        # Try Redis first if available
        if self._redis:
            try:
                self._redis.delete(key)
            except:
                pass
        
        # Also remove from local cache
        self._local_cache.pop(key, None)
        return True
    
    async def flush(self) -> bool:
        """Clear all cache entries"""
        if self._redis:
            try:
                self._redis.flushall()
            except:
                pass
        
        self._local_cache.clear()
        return True 