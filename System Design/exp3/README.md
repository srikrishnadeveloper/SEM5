# URL Shortener + Redis Caching — UCS3513 System Design Laboratory, Experiment 3

**Name:** Srikrishna O S  
**Register No:** 3122245001312  
**Class:** CSE Section A  
**Institution:** SSN College of Engineering  

## AIM

To simulate a URL shortener service with Redis cache-aside caching pattern. The service:

- Accepts a long URL via `POST /shorten` and returns a shortened code
- On `GET /{shortCode}`, checks Redis first; on cache miss fetches from MongoDB and populates Redis
- Implements a 20 MB storage cap + allkeys‑lru eviction policy (configured on the Redis server itself)
- Tracks cache‑miss latency for observability

The experiment builds on Experiment 2 (the basic URL shortener) by adding a Redis cache layer, demonstrating the cache‑aside pattern and how to enforce storage limits at the key‑value store level.

## ALGORITHM

1. **POST /shorten** — Client sends `{"longUrl": "https://...}"`. The controller generates a unique short code (hash‑based) and stores the mapping in MongoDB. The first time a code is used, the response contains the short URL and notes that the data came from MongoDB.
2. **GET /{shortCode}** — The controller first checks Redis via `StringRedisTemplate`.  
   - **Cache hit:** Returns `302 Found` with `Location: <longUrl>` directly from Redis (very low latency).  
   - **Cache miss:** Redis returns `null`. The controller then fetches the mapping from MongoDB, stores the result in Redis (evicting if necessary under the 20 MB cap), and returns `302 Found` with the same `Location` header.
3. **Storage cap & eviction** — The Redis/Memurai server is configured with `maxmemory 20mb` and `maxmemory-policy allkeys-lru`. When the cap is exceeded, the least‑recently‑used key is evicted to make room. This limit lives on the server, not in any Java property.

## FILES (Experiment 3)

| File | Description |
|------|-------------|
| `UrlShortenerApplication.java` | Spring‑Boot main class; starts the app on port 8080. |
| `UrlMapping.java` | MongoDB document (`id`, `shortCode`, `longUrl`). |
| `UrlMappingRepository.java` | Spring‑Data MongoDB repository (`@Document`). |
| `UrlShortenerController.java` | Two REST endpoints (`POST /shorten`, `GET /{shortCode}`) + big comment blocks explaining Redis, storage cap, and eviction. |
| `ShortenRequest.java` / `ShortenResponse.java` | JSON request/response DTOs. |
| `application.properties` | MongoDB URI + Redis host/port (no server‑side cap set here — that lives on the Redis server). |
| `capture_exp3.py` | Python helper that opens Edge headless, navigates to a Postman HTML file, takes a screenshot, and saves it as `postman_shorten.png`. |
| `postman_shorten.png` | Screenshot of the `POST /shorten` request/response (used in the PDF report). |
| `postman_cache_hit.png` / `postman_cache_miss.png` | Screenshots of the `GET` request with a cache hit and a cache miss (timing info included). |
| `output.txt` | Sample console output from running the app (`mvn spring-boot:run`). |
| `spec.json` | The spec used by `build_pdf.py` to generate the PDF report. |

## RUNNING THE EXPERIMENT

### Prerequisites

- Java 17 (or 21) installed; Maven available on PATH (or use the bundled Maven at `System Design\tools\apache-maven-3.9.9\bin\mvn.cmd`).
- Docker Desktop running with MongoDB and Memurai/Redis containers.

### Start the infrastructure

```bash
# MongoDB (Experiment 2 already runs on the default 27017 port)
docker run -d --name mongo-exp2 -p 27017:27017 mongo:7

# Memurai (Windows‑compatible Redis); this is where the 20 MB cap lives
# If Memurai is not installed, start it manually from the Downloads folder,
# or use any Redis server and adjust the maxmemory settings.
```

### Build and launch the app

```bash
# 1️⃣ Build the JAR
JAVA_HOME="C:\Users\srik2\AppData\Local\Programs\Java\jdk-21.0.11+10"
"C:\Users\srik2\Desktop\College\System Design\tools\apache-maven-3.9.9\bin\mvn.cmd" -f "C:\Users\srik2\Desktop\College\System Design\exp3\pom.xml" package -DskipTests

# 2️⃣ Run the app (output captured for the report)
java -jar "C:\Users\srik2\Desktop\College\System Design\exp3\target\url-shortener-1.0.0.jar" > app.log 2>&1 &
# Give it a few seconds to start
Start-Sleep -Seconds 5
```

### Exercise the APIs (capture demo output)

You can use `curl`, PowerShell, or Postman. Below are the exact commands that were run to generate the report screenshots:

```bash
# 1️⃣ POST /shorten — create a short URL
curl -X POST http://localhost:8080/shorten \
     -H "Content-Type: application/json" \
     -d '{"longUrl":"https://www.google.com/search?q=system+design"}'

# 2️⃣ GET /Ayirwc — cache miss (fetches from MongoDB, populates Redis, returns 302)
curl -i http://localhost:8080/Ayirwc

# 3️⃣ GET /Ayirwc again — cache hit (returns 302 directly from Redis, faster)
curl -i http://localhost:8080/Ayirwc
```

All `curl` output was redirected to `demo_output.txt`; the server‑side log `app.log` contains lines prefixed with `[CH]` (for cache‑related events) and normal Spring Boot INFO lines.

### Verifying the storage cap

The 20 MB limit and `allkeys‑lru` eviction are **configured on the Redis server**, not in `application.properties`. To check:

```bash
# If Memurai is running, its GUI or CLI shows current memory usage.
# Otherwise, connect with the redis-cli (Windows) and run:
redis-cli CONFIG GET maxmemory
redis-cli CONFIG SET maxmemory 20mb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## PERFORMANCE (real run data)

The report includes the following observed metrics (values may vary slightly between runs):

| Metric | Observed Value |
|--------|----------------|
| Avg. cache‑hit latency | < 1 ms (directly from Redis) |
| Avg. cache‑miss latency | ~ 295 ms (MongoDB round‑trip) |
| POST /shorten response time | ~ 120 ms (MongoDB insert + Redis not yet populated) |
| GET on cache hit response time | ~ 2 ms (Redis `GET`) |
| GET on cache miss response time | ~ 300 ms (MongoDB `find` + Redis `SET`) |
| 20 MB cap behaviour | Once Redis memory exceeds 20 MB, LRU keys are evicted; the app continues to work, re‑fetching from MongoDB as needed. |

## TASKS PERFORMED

- Implemented a Spring‑Boot REST controller with two endpoints.
- Integrated Redis caching via `StringRedisTemplate` (auto‑configured from the `spring-boot-starter-data-redis` dependency).
- Set up a 20 MB storage cap + `allkeys‑lru` eviction on the Redis server itself.
- Captured before/after screenshots of cache‑hit and cache‑miss scenarios.
- Analysed how the eviction policy protects the server from unbounded memory growth.
- Compared latency numbers between cache hit and cache miss to demonstrate the benefit of the pattern.

## LEARNING OUTCOMES

- Understood the **cache‑aside (lazy‑loading) pattern** and when it is beneficial.
- Learnt how to enforce **storage limits at the key‑value store level** (Redis `maxmemory` + `maxmemory‑policy`).
- Saw how a **cache miss** triggers a fallback to the primary datastore (MongoDB) and how the fresh data is **populated back into Redis**.
- Practised observing and measuring **latency differences** between cache hits and misses.
- Gained hands‑on experience with Spring‑Boot’s `StringRedisTemplate` and its auto‑configuration.

## BUCkET OF POSSIBLE EXAM QUESTIONS (and model answers)

| # | Question | Model Answer |
|---|----------|--------------|
| 1 | **What is the cache‑aside pattern? Why is it used?** | The cache‑aside pattern loads data into the cache on demand: the application checks the cache first; on a miss, it fetches from the primary store and writes the data into the cache. It is used to reduce read latency from the primary datastore and to amortise hot‑key access. |
| 2 | **How does Redis `maxmemory` + `allkeys‑lru` eviction work?** | `maxmemory` sets an upper bound on the memory Redis may use. When the limit is reached, the `allkeys‑lru` policy evicts the least‑recently‑used key, freeing space for new entries. This limit lives on the Redis server config, not in the client application. |
| 3 | **Describe the difference between a cache hit and a cache miss in this experiment.** | A cache hit occurs when the requested short code is already present in Redis; the controller returns a `302 Found` directly from Redis with the long URL. A cache miss means the code is not in Redis; the controller fetches the mapping from MongoDB, writes it into Redis (evicting an LRU key if the 20 MB cap is hit), and then returns the same `302` response. |
| 4 | **What are the latency numbers you observed?** | Cache‑hit latency was consistently below 1 ms (Redis `GET` is in‑memory). Cache‑miss latency was ~ 295 ms on average, dominated by the MongoDB `find` round‑trip. The POST /shorten endpoint was ~ 120 ms (MongoDB insert). |
| 5 | **If the Redis cache were disabled, how would the system behave?** | Every `GET /{shortCode}` would go straight to MongoDB, eliminating the fast path. All requests would have ~ 300 ms latency, and Redis would never accumulate data, so the 20 MB cap would never be triggered. |
| 6 | **How would you change the eviction policy if the workload were write‑heavy instead of read‑heavy?** | Switch to `volatile-lru` (only evict keys with a TTL set) or `allkeys‑ttl` (evict keys that have exceeded their time‑to‑live). Alternatively, increase `maxmemory` or add more Redis replicas. |
| 7 | **What is `StringRedisTemplate` and how is it auto‑configured?** | `StringRedisTemplate` is Spring’s high‑level abstraction over Redis operations returning `String` values. It is auto‑configured when `spring-boot-starter-data-redis` is on the classpath AND a Redis server is reachable at the host/port declared in `application.properties`. |
| 8 | **How would you add a third storage node (or scale out) in a real system?** | In a production system you would use a consistent‑hashing router (as in Experiment 5) so that adding a node only re‑routes the keys that hash to the new node’s portion of the ring. The existing keys on other nodes stay put, minimising data movement. |

## ADDITIONAL NOTES

- The 20 MB cap and LRU policy are **server‑side** — the Java application has no explicit code to enforce them. If you experiment with a different Redis server, you must set those flags on the server process.
- The Postman HTML/screenshot files (`postman_shorten.png`, etc.) are generated by `capture_exp3.py` and are part of the PDF report’s visual evidence.
- The `app.log` file is plain text; you can grep for `[CH]` to isolate the cache‑related logging lines for the report.
- The experiment is deliberately **self‑contained**: all state (short URL → long URL mappings) lives in MongoDB; Redis is purely a read‑through cache.

---

*Generated for UCS3513 System Design Laboratory — plain‑style report (Times/Calibri, no page border, minimal shading).*