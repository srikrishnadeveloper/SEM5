local tokensKey = KEYS[1]
local lastRefillKey = KEYS[2]

local capacity = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])
local amount = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens = tonumber(redis.call('GET', tokensKey)) or capacity
local lastRefill = tonumber(redis.call('GET', lastRefillKey)) or now

-- Refill tokens based on elapsed time
local elapsed = now - lastRefill
if elapsed >= interval then
    local intervals = math.floor(elapsed / interval)
    tokens = math.min(capacity, tokens + (intervals * amount))
    lastRefill = lastRefill + (intervals * interval)
end

-- Consume 1 token if available
if tokens > 0 then
    tokens = tokens - 1
    redis.call('SET', tokensKey, tostring(tokens))
    redis.call('SET', lastRefillKey, tostring(lastRefill))
    return 1 -- Success (Allowed)
end

redis.call('SET', tokensKey, tostring(tokens))
redis.call('SET', lastRefillKey, tostring(lastRefill))
return 0 -- Failed (Throttled)