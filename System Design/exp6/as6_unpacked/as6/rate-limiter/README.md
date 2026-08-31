# API Rate Limiter using Token Bucket Algorithm

Spring Boot + Redis implementation of the Token Bucket algorithm for API rate limiting.

## Project structure

```
rate-limiter/
├── pom.xml
├── README.md
├── test-rate-limiter.sh          # Bash test script (Linux/Mac/WSL)
├── test-rate-limiter.bat         # Batch test script (Windows CMD)
└── src/main/
    ├── java/com/ssn/ratelimiter/
    │   ├── RateLimiterApplication.java   (main Spring Boot class)
    │   ├── TokenBucketService.java       (core token bucket logic)
    │   └── RateLimiterController.java    (REST endpoints)
    └── resources/
        └── application.properties        (Redis + bucket config)
```

## How the algorithm works

- Every client (identified by a `client-id` header, or IP address if not
  given) has its own **bucket** of tokens, stored in Redis as two keys:
  `rate_limit:<client>:tokens` and `rate_limit:<client>:last_refill`.
- The bucket starts full, at `bucket-capacity` tokens.
- Each request to `/api/resource` tries to consume **1 token**.
  - If a token is available → request succeeds (HTTP 200), 1 token is deducted.
  - If no tokens are left → request is rejected (HTTP 429 Too Many Requests).
- Tokens refill automatically over time: every `refill-interval-millis`
  milliseconds, `refill-amount` tokens are added back, up to the capacity.
- Storing the bucket state in **Redis** (rather than a local `HashMap`) means
  the rate limit is centralized — if you ran multiple copies of this app
  behind a load balancer, they'd all share the same bucket per client instead
  of each having their own independent limit.
- The `tryConsume` method is `synchronized` so that "read tokens → refill →
  decrement" happens as one atomic step, even if multiple requests from the
  same client arrive at once. (A Redis Lua script is a more advanced way to
  get the same atomicity across multiple app instances, but `synchronized` is
  enough to explain and demonstrate for a single-instance lab setup.)

## Prerequisites

- Java 17+
- Maven
- Redis running on `localhost:6379`

| Platform | Redis option |
|----------|-------------|
| Linux/Mac | `redis-server` or Docker |
| Windows | Docker Desktop (`docker run -d --name redis-ratelimiter -p 6379:6379 redis:7`) |

## Run

### Linux / Mac / WSL

```bash
mvn spring-boot:run
```

### Windows (CMD / PowerShell)

```cmd
mvn spring-boot:run
```

Maven must be in your system PATH. If installed via NetBeans, add
`C:\Program Files\Apache NetBeans\java\maven\bin` to your PATH.

The app starts on `http://localhost:8080`.

## Endpoints

| Method | Path                    | Description                                  |
|--------|-------------------------|-----------------------------------------------|
| GET    | `/api/health`           | Health check (not rate limited)              |
| GET    | `/api/resource`         | Rate-limited resource                        |
| GET    | `/api/rate-limit-status`| Check remaining tokens without consuming one |

## Testing

### Linux / Mac / WSL (bash)

**Normal traffic** (should return 200):
```bash
curl -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Burst traffic** (send more requests than the bucket capacity quickly to see 429s):
```bash
for i in {1..15}; do curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/api/resource -H "client-id: user1"; done
```

**Check remaining tokens:**
```bash
curl -i http://localhost:8080/api/rate-limit-status -H "client-id: user1"
```

**Wait and retry:**
```bash
sleep 5
curl -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Run the test script:**
```bash
chmod +x test-rate-limiter.sh
./test-rate-limiter.sh          # 15 requests, client-id "user1"
./test-rate-limiter.sh 20       # 20 requests
./test-rate-limiter.sh 20 user2 # 20 requests, client-id "user2"
```

### Windows (CMD / PowerShell)

> **Note:** In CMD use `curl.exe` (not `curl`, which is a PowerShell alias for `Invoke-WebRequest`).

**Normal traffic** (should return 200):
```cmd
curl.exe -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Check remaining tokens:**
```cmd
curl.exe -i http://localhost:8080/api/rate-limit-status -H "client-id: user1"
```

**Wait and retry** (use `timeout` instead of `sleep`):
```cmd
timeout /t 5 /nobreak >nul
curl.exe -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Run the test script:**
```cmd
test-rate-limiter.bat              # 15 requests, client-id "user1"
test-rate-limiter.bat 20           # 20 requests
test-rate-limiter.bat 20 user2     # 20 requests, client-id "user2"
```

With the default config (`bucket-capacity=10`), you should see `200` for the
first 10 requests and `429` for the rest, since tokens refill only 1 per
second.

## Points to explain in viva

1. **Token Bucket vs other algorithms** — unlike a fixed window counter, the
   token bucket allows short bursts (up to the bucket capacity) while still
   enforcing a long-term average rate (the refill rate).
2. **Why Redis** — centralizes rate-limit state so it works correctly even
   if the API is scaled horizontally across multiple instances.
3. **Bucket capacity vs refill rate** — capacity controls the maximum burst
   size; refill rate controls the sustained (steady-state) request rate.
4. **Concurrency** — `synchronized` ensures no two threads can read stale
   token counts and both "succeed" when only one token remains.
5. **HTTP 429** — the standard status code for "Too Many Requests", used to
   signal clients they've been rate limited.
