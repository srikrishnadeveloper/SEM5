package com.ssn.ratelimiter;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class RateLimiterController {

    private final TokenBucketService tokenBucketService;

    public RateLimiterController(TokenBucketService tokenBucketService) {
        this.tokenBucketService = tokenBucketService;
    }

    /** Health check endpoint - intentionally NOT rate limited. */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Service is healthy");
    }

    /**
     * Rate-limited endpoint. Client is identified by the "client-id" header
     * if present, otherwise falls back to the request's IP address.
     */
    @GetMapping("/resource")
    public ResponseEntity<String> resource(
            @RequestHeader(value = "client-id", required = false) String clientId,
            HttpServletRequest request) {

        String identifier = resolveClientId(clientId, request);

        if (tokenBucketService.tryConsume(identifier)) {
            return ResponseEntity.ok("Request processed successfully");
        } else {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .body("HTTP 429 Too Many Requests: rate limit exceeded");
        }
    }

    /** Lets you check remaining tokens without consuming one. */
    @GetMapping("/rate-limit-status")
    public ResponseEntity<String> rateLimitStatus(
            @RequestHeader(value = "client-id", required = false) String clientId,
            HttpServletRequest request) {

        String identifier = resolveClientId(clientId, request);
        long tokens = tokenBucketService.getRemainingTokens(identifier);
        return ResponseEntity.ok("Remaining tokens: " + tokens);
    }

    private String resolveClientId(String clientId, HttpServletRequest request) {
        if (clientId != null && !clientId.isBlank()) {
            return clientId;
        }
        String ip = request.getRemoteAddr();
        return (ip != null) ? ip : "unknown";
    }
}
