package com.ssn.urlshortener;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RateLimiterApplication {

    public static void main(String[] args) {
        SpringApplication.run(RateLimiterApplication.class, args);
    }
}

package com.ssn.urlshortener;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping("/api")
public class RateLimiterController {

    @Autowired
    private StringRedisTemplate redisTemplate;

    private final int bucketCapacity;
    private final long refillIntervalMillis;
    private final long refillAmount;

    private static final String RATE_LIMIT_KEY_PREFIX = "rate_limit:";

    public RateLimiterController(@Value("${rate.limiter.bucket-capacity:100}") int bucketCapacity,
                                 @Value("${rate.limiter.refill-interval-millis:1000}") long refillIntervalMillis,
                                 @Value("${rate.limiter.refill-amount:1}") long refillAmount) {
        this.bucketCapacity = bucketCapacity;
        this.refillIntervalMillis = refillIntervalMillis;
        this.refillAmount = refillAmount;
    }

    private String key(String clientId) {
        return RATE_LIMIT_KEY_PREFIX + clientId;
    }

    private long getLastRefillTime(String clientId) {
        return redisTemplate.opsForValue().get(key(clientId) + ":last") != null
                ? Long.parseLong(redisTemplate.opsForValue().get(key(clientId) + ":last"))
                : System.currentTimeMillis();
    }

    private boolean tryConsumeToken(String clientId) {
        String key = key(clientId);
        long now = System.currentTimeMillis();

        String tokensStr = redisTemplate.opsForValue().get(key);
        long tokens = (tokensStr != null) ? Long.parseLong(tokensStr) : bucketCapacity;

        long elapsed = now - getLastRefillTime(clientId);
        long refillCount = (int) (elapsed / refillIntervalMillis);
        if (refillCount > 0) {
            tokens = Math.min(bucketCapacity, tokens + refillCount * refillAmount);
            redisTemplate.opsForValue().set(key, String.valueOf(tokens), refillIntervalMillis, TimeUnit.MILLISECONDS);
        }

        if (tokens > 0) {
            redisTemplate.opsForValue().decrement(key);
            return true;
        }
        return false;
    }

    // Endpoint without rate limiting (for health checks)
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Service is healthy");
    }

    // Rate-limited endpoint
    @GetMapping("/resource")
    public ResponseEntity<String> resource(@RequestHeader(value = "client-id", required = false) String clientId) {
        String identifier = (clientId != null) ? clientId : "unknown";

        if (tryConsumeToken(identifier)) {
            return ResponseEntity.ok("Request processed successfully");
        } else {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .body("HTTP 429 Too Many Requests: rate limit exceeded");
        }
    }

    // Get current rate limit status
    @GetMapping("/rate-limit-status")
    public ResponseEntity<String> rateLimitStatus(@RequestHeader(value = "client-id", required = false) String clientId) {
        String identifier = (clientId != null) ? clientId : "unknown";
        String key = key(identifier);
        long tokens = (redisTemplate.opsForValue().get(key) != null)
                ? Long.parseLong(redisTemplate.opsForValue().get(key))
                : bucketCapacity;
        return ResponseEntity.ok("Remaining tokens: " + tokens);
    }

    private String getClientIp() {
        return "unknown";
    }
}