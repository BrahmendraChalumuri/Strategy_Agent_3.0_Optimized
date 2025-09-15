"""
Rate Limiting Configuration for Perplexity API

Based on Perplexity API documentation:
- Default rate limit: 50 requests per minute for 'sonar' model
- Rate limits vary by usage tier (based on total credits purchased)
- Tier 0: $0 in total credits
- Tier 1: $50 in total credits  
- Tier 2: $250 in total credits
- Tier 3: $500 in total credits
- Tier 4: $1,000 in total credits
- Tier 5: $5,000 in total credits

Conservative settings to avoid 429 errors:
"""

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    # Conservative settings to avoid 429 errors
    "max_requests_per_minute": 30,  # Well below the 50/min limit
    "max_concurrent_requests": 5,   # Limit concurrent connections
    
    # Exponential backoff settings
    "base_retry_delay": 1,          # Base delay in seconds
    "max_retry_delay": 60,          # Maximum delay in seconds
    "retry_multiplier": 2,          # Exponential backoff multiplier
    "max_retries": 3,               # Maximum number of retries
    
    # Jitter settings to prevent thundering herd
    "jitter_min": 0.1,              # Minimum jitter in seconds
    "jitter_max": 0.5,              # Maximum jitter in seconds
    
    # Connection pool settings
    "connection_pool_size": 5,      # Total connection pool size
    "connections_per_host": 3,      # Connections per host
}

# Usage tier configurations (for reference)
USAGE_TIERS = {
    "tier_0": {
        "total_credits": 0,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 5
    },
    "tier_1": {
        "total_credits": 50,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 10
    },
    "tier_2": {
        "total_credits": 250,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 20
    },
    "tier_3": {
        "total_credits": 500,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 30
    },
    "tier_4": {
        "total_credits": 1000,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 40
    },
    "tier_5": {
        "total_credits": 5000,
        "sonar_rate_limit": 50,
        "sonar_deep_research_rate_limit": 50
    }
}

def get_rate_limit_config():
    """Get the current rate limit configuration"""
    return RATE_LIMIT_CONFIG

def update_rate_limits(requests_per_minute=None, concurrent_requests=None):
    """Update rate limits dynamically"""
    if requests_per_minute is not None:
        RATE_LIMIT_CONFIG["max_requests_per_minute"] = requests_per_minute
    if concurrent_requests is not None:
        RATE_LIMIT_CONFIG["max_concurrent_requests"] = concurrent_requests
    
    return RATE_LIMIT_CONFIG
