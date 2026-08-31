package com.ssn.ratelimiter;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

/**
 * Implements the Token Bucket algorithm for API rate limiting.
 *
 * Each client (identified by an ID or IP) gets its own "bucket" of tokens,
 * stored in Redis so the state is shared across all requests / instances.
 *
 * - The bucket starts full (bucketCapacity tokens).
 * - Every request tries to consume 1 token.
 * - Tokens are refilled at a fixed rate over time (refillAmount tokens
 *   every refillIntervalMillis milliseconds), up to the bucket capacity.
 * - If no tokens are available, the request is rejected (HTTP 429).
 *
 * Redis is used here (instead of a plain in-memory HashMap) so that the
 * rate-limit state is centralized: if this application were scaled to
 * multiple instances behind a load balancer, all instances would still
 * share the same bucket for a given client.
 */

@Service
public class TokenBucketService {

    private final StringRedisTemplate redisTemplate;

    private final int bucketCapacity;
    private final long refillIntervalMillis;
    private final long refillAmount;

    private static final String TOKENS_SUFFIX = ":tokens";
    private static final String LAST_REFILL_SUFFIX = ":last_refill";
    private static final String KEY_PREFIX = "rate_limit:";

    public TokenBucketService(
            StringRedisTemplate redisTemplate,
            @Value("${rate.limiter.bucket-capacity:10}") int bucketCapacity,
            @Value("${rate.limiter.refill-interval-millis:1000}") long refillIntervalMillis,
            @Value("${rate.limiter.refill-amount:1}") long refillAmount) {
        this.redisTemplate = redisTemplate;
        this.bucketCapacity = bucketCapacity;
        this.refillIntervalMillis = refillIntervalMillis;
        this.refillAmount = refillAmount;
    }

    /**
     * Tries to consume one token for the given client.
     * Returns true if the request is allowed, false if the client is rate limited.
     *
     * synchronized is used here (instead of a Redis Lua script) to keep the
     * read -> refill -> decrement sequence atomic. This is enough for a
     * single-instance lab setup and is easy to explain: only one thread at a
     * time can update a given client's bucket.
     */
    public synchronized boolean tryConsume(String clientId) {
        refillBucket(clientId);

        long tokens = getTokens(clientId);
        if (tokens > 0) {
            redisTemplate.opsForValue().decrement(tokensKey(clientId));
            return true;
        }
        return false;
    }

    /** Returns the current token count for a client, applying any pending refill first. */
    public long getRemainingTokens(String clientId) {
        refillBucket(clientId);
        return getTokens(clientId);
    }

    // ---- internal helpers ----   

    private void refillBucket(String clientId) {
        long now = System.currentTimeMillis();
        String lastRefillStr = redisTemplate.opsForValue().get(lastRefillKey(clientId));

        // First time we see this client: initialize a full bucket.
        if (lastRefillStr == null) {
            redisTemplate.opsForValue().set(tokensKey(clientId), String.valueOf(bucketCapacity));
            redisTemplate.opsForValue().set(lastRefillKey(clientId), String.valueOf(now));
            return;
        }

        long lastRefillTime = Long.parseLong(lastRefillStr);
        long elapsed = now - lastRefillTime;

        long intervalsPassed = elapsed / refillIntervalMillis;
        if (intervalsPassed <= 0) {
            return; // not time to refill yet
        }

        long currentTokens = getTokens(clientId);
        long newTokens = Math.min(bucketCapacity, currentTokens + (intervalsPassed * refillAmount));

        redisTemplate.opsForValue().set(tokensKey(clientId), String.valueOf(newTokens));
        // advance the "last refill" timestamp only by the whole intervals consumed,
        // so partial time isn't lost on the next check
        redisTemplate.opsForValue().set(lastRefillKey(clientId),
                String.valueOf(lastRefillTime + intervalsPassed * refillIntervalMillis));
    }

    

    private long getTokens(String clientId) {
        String tokensStr = redisTemplate.opsForValue().get(tokensKey(clientId));
        return (tokensStr != null) ? Long.parseLong(tokensStr) : bucketCapacity;
    }

    private String tokensKey(String clientId) {
        return KEY_PREFIX + clientId + TOKENS_SUFFIX;
    }

    private String lastRefillKey(String clientId) {
        return KEY_PREFIX + clientId + LAST_REFILL_SUFFIX;
    }
}

// ├── "rate_limit:user123:tokens"      ──> "8"              (String)
// └── "rate_limit:user123:last_refill"  ──> "1772102069000"  (String)
