package com.ssn.ratelimiter;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TokenBucketService {

    private final StringRedisTemplate redisTemplate;

    private final int bucketCapacity;
    private final long refillIntervalMillis;
    private final long refillAmount;

    private static final String TOKENS_SUFFIX = ":tokens";
    private static final String LAST_REFILL_SUFFIX = ":last_refill";
    private static final String KEY_PREFIX = "rate_limit:";

    private final DefaultRedisScript<Long> rateLimitScript;

    public TokenBucketService(
            StringRedisTemplate redisTemplate,
            @Value("${rate.limiter.bucket-capacity:10}") int bucketCapacity,
            @Value("${rate.limiter.refill-interval-millis:1000}") long refillIntervalMillis,
            @Value("${rate.limiter.refill-amount:1}") long refillAmount) {
        this.redisTemplate = redisTemplate;
        this.bucketCapacity = bucketCapacity;
        this.refillIntervalMillis = refillIntervalMillis;
        this.refillAmount = refillAmount;

        // Load the Lua script from classpath resources
        this.rateLimitScript = new DefaultRedisScript<>();
        this.rateLimitScript.setLocation(new ClassPathResource("scripts/rate_limiter.lua"));
        this.rateLimitScript.setResultType(Long.class);
    }

    /**
     * Tries to consume one token using an atomic Redis Lua script.
     * Replaces Java 'synchronized' to support distributed scaling safely.
     */
    public boolean tryConsume(String clientId) {
        Long result = redisTemplate.execute(
        rateLimitScript, // 1. The script to run
        
        // 2. KEYS (Passed into Lua as KEYS[1] and KEYS[2])
        List.of(
            tokensKey(clientId),      // -> "rate_limit:user1:tokens"  (KEYS[1])
            lastRefillKey(clientId)   // -> "rate_limit:user1:last_refill" (KEYS[2])
        ),
        
        // 3. ARGV (Passed into Lua as ARGV[1], ARGV[2], ARGV[3], ARGV[4])
        String.valueOf(bucketCapacity),           // -> "10"    (ARGV[1])
        String.valueOf(refillIntervalMillis),     // -> "1000"  (ARGV[2])
        String.valueOf(refillAmount),             // -> "1"     (ARGV[3])
        String.valueOf(System.currentTimeMillis())// -> "1772102069000" (ARGV[4])
    );  

        return result != null && result == 1L;
    }

    /** Returns current token count without consuming. */
    public long getRemainingTokens(String clientId) {
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