local tokensKey = KEYS[1]
local lastRefillKey = KEYS[2]
local capacity = tonumber(ARGV[1])
local refillInterval = tonumber(ARGV[2])
local refillAmount = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Fetch current state from Redis
local tokens = tonumber(redis.call('GET', tokensKey))
local lastRefill = tonumber(redis.call('GET', lastRefillKey))

-- Initialize for first-time clients
if not tokens or not lastRefill then
    tokens = capacity
    lastRefill = now
else
    -- Refill logic based on elapsed time
    local elapsed = now - lastRefill
    if elapsed >= refillInterval then
        local intervalsPassed = math.floor(elapsed / refillInterval)
        tokens = math.min(capacity, tokens + (intervalsPassed * refillAmount))
        lastRefill = lastRefill + (intervalsPassed * refillInterval)
    end
end

-- Consume token if available
if tokens > 0 then
    tokens = tokens - 1
    redis.call('SET', tokensKey, tostring(tokens))
    redis.call('SET', lastRefillKey, tostring(lastRefill))
    return 1 -- Allowed
else
    redis.call('SET', tokensKey, tostring(tokens))
    redis.call('SET', lastRefillKey, tostring(lastRefill))
    return 0 -- Rejected (Rate Limited)
end