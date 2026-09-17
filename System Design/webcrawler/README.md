# UCS3513 System Design Laboratory — Lab Exercise 8
## Web Crawler for Automated Web Content Discovery

Spring Boot + Redis based web crawler with a REST API, Docker deployment, and a JMeter load-test plan.

---

## 1. Project Structure

```
webcrawler/
├── pom.xml
├── Dockerfile
├── docker-compose.yml
├── dataset/
│   └── seed-urls.txt                 # sample seed URLs
├── jmeter/
│   └── crawler_load_test.jmx         # JMeter test plan
└── src/main/
    ├── java/com/sscse/webcrawler/
    │   ├── WebcrawlerApplication.java
    │   ├── config/
    │   │   ├── RedisConfig.java          # RedisTemplate bean
    │   │   └── CrawlerExecutorConfig.java# thread pool for async crawling
    │   ├── model/                        # CrawlRequest, PageCrawlResult, enums, status DTO
    │   ├── service/
    │   │   ├── RedisCrawlStore.java       # queue / visited-set / page-cache / stats in Redis
    │   │   └── CrawlerService.java        # BFS crawl algorithm, link extraction, validation
    │   ├── controller/
    │   │   ├── CrawlerController.java     # REST API
    │   │   └── GlobalExceptionHandler.java
    │   └── util/UrlValidator.java         # URL validation & normalization
    └── resources/
        ├── application.properties
        └── static/sample/page1..5.html    # local interlinked test pages (no internet needed)
```

---

## 2. How It Works

| Component | Redis structure | Purpose |
|---|---|---|
| URL Queue | `List` — `crawler:queue` | FIFO of URLs waiting to be crawled (`LPUSH` / `RPOP`) |
| Visited-URL Set | `Set` — `crawler:visited` | O(1) duplicate check (`SADD` / `SISMEMBER`) |
| Page metadata cache | `String` (JSON) — `crawler:page:{url}` | title, status, discovered links, timestamp |
| Stats counters | `String` — `crawler:stats:*` | atomic `INCR` for crawled/failed counts |

**Flow:** seed URL → validated → pushed to queue → worker threads pop URLs → check visited set →
fetch with Jsoup → extract `<a href>` links → validate/normalize → enqueue new links → repeat until
`maxPages`/`maxDepth` budget is exhausted or the queue empties.

---

## 3. Running Locally (without Docker)

```bash
# 1. Start Redis
redis-server        # or: docker run -p 6379:6379 redis:7-alpine

# 2. Build & run the app
cd webcrawler
mvn clean package -DskipTests
java -jar target/webcrawler.jar

# App starts on http://localhost:8080
```

## 4. Running with Docker (app + Redis)

```bash
cd webcrawler
docker compose up --build
```

This builds the Spring Boot image, starts Redis, and wires them together via
`docker-compose.yml` (the app connects to Redis using the `redis` service hostname).

---

## 5. Testing the REST API (Postman / curl)

**Start a crawl** (use the bundled local sample site — no internet required):
```bash
curl -X POST http://localhost:8080/api/crawler/start \
  -H "Content-Type: application/json" \
  -d '{
        "seedUrl": "http://localhost:8080/sample/page1.html",
        "maxDepth": 2,
        "maxPages": 10,
        "sameDomainOnly": true
      }'
```

**Check status:**
```bash
curl http://localhost:8080/api/crawler/status
```

**View visited URLs:**
```bash
curl http://localhost:8080/api/crawler/visited
```

**Peek the queue:**
```bash
curl http://localhost:8080/api/crawler/queue?limit=20
```

**Get all crawl results (titles, links found, status per page):**
```bash
curl http://localhost:8080/api/crawler/results
```

**Reset all crawler state:**
```bash
curl -X DELETE http://localhost:8080/api/crawler/reset
```

Import these as a Postman collection by creating requests for each endpoint above, or use
Postman's "Import from raw text" with the curl commands.

---

## 6. Using a Real Public Seed URL

Once you've verified the flow with the local sample pages, try public sites listed in
`dataset/seed-urls.txt`, e.g.:
```bash
curl -X POST http://localhost:8080/api/crawler/start \
  -H "Content-Type: application/json" \
  -d '{"seedUrl":"https://books.toscrape.com","maxDepth":1,"maxPages":15,"sameDomainOnly":true}'
```

---

## 7. Load Testing with JMeter

1. Open `jmeter/crawler_load_test.jmx` in the JMeter GUI (or run headless).
2. It defines a Thread Group of **50 concurrent users**, ramp-up **10 s**, each firing one
   `POST /api/crawler/start` followed by a `GET /api/crawler/status` call.
3. Run headless and generate an HTML dashboard:
   ```bash
   jmeter -n -t jmeter/crawler_load_test.jmx -l results.jtl -e -o report/
   ```
4. Vary **Number of Threads (users)** — try 10, 50, 100, 200 — and record:
   - Average / 90th-percentile response time
   - Throughput (requests/sec)
   - Error % (connection refused, timeouts)
5. Because the crawl runs asynchronously on a fixed thread pool
   (`crawler.thread-pool-size` in `application.properties`), increasing concurrent HTTP
   requests beyond the pool size will queue crawl jobs rather than fail — a good discussion
   point for the performance analysis.

---

## 8. Analysis — Role of Each Component

- **Seed URL**: The starting point supplied by the user; it is the first entry pushed onto the
  URL queue and anchors the domain-scope check (`sameDomainOnly`) used to decide which
  discovered links are worth following.

- **URL Queue**: A FIFO structure (implemented as a Redis List) that holds every URL waiting
  to be fetched. It decouples *discovery* (finding new links) from *processing* (fetching a
  page), enabling breadth-first exploration and allowing multiple worker threads to pull work
  concurrently without scanning any other data structure.

- **Visited-URL Set**: A Redis Set that records every URL already processed. Because Set
  membership checks (`SISMEMBER`) are O(1), the crawler can instantly tell whether a newly
  discovered link is worth queuing, which is what prevents infinite loops on cyclic link
  graphs and eliminates redundant network calls.

- **Web Crawler (CrawlerService)**: The orchestrator that ties the queue, the visited set, and
  the HTML fetch/parse step together — it dequeues a URL, validates it, fetches the page with
  Jsoup, extracts `<a href>` links, and re-enqueues the ones that are new and within the
  depth/page budget.

- **Redis Cache**: Serves three roles simultaneously — (1) the durable, shareable backing
  store for the queue and visited set so state survives restarts and could be shared across
  multiple crawler instances; (2) a metadata cache (`crawler:page:{url}`) so a repeated lookup
  of a page's crawl result doesn't require re-fetching it; (3) atomic counters for crawled/
  failed stats, useful for the `/status` endpoint and JMeter analysis.

### Analysis Questions

**1. How does the crawler discover new URLs from a retrieved web page?**
After fetching a page with Jsoup, the crawler selects every `a[href]` element, resolves each
`href` to an absolute URL (`abs:href`), and collects them as candidate links (`extractLinks()`
in `CrawlerService`).

**2. How does the system determine whether a newly discovered URL should be crawled?**
Each candidate link passes through `UrlValidator.isValid()` (correct scheme, non-blank host,
not a disallowed binary extension), is normalized (fragment/trailing-slash stripped), optionally
checked against the seed's domain if `sameDomainOnly=true`, and checked against the current
depth budget (`depth < maxDepth`). Only links that pass all of these are enqueued.

**3. How does the system process a URL that has not been visited?**
`store.markVisitedIfAbsent(url)` atomically adds it to the Redis visited Set and returns `true`.
The crawler then fetches the page, parses its title and links, stores a `PageCrawlResult` in
Redis, increments the crawled counter, and enqueues any newly discovered links.

**4. How does the system process a URL that has already been visited?**
`markVisitedIfAbsent()` returns `false` (the Set add was a no-op), so the crawler immediately
skips the URL — no HTTP request is made and nothing is re-queued, preventing duplicate work.

**5. Compare queue-based URL crawling with repeatedly scanning the complete collection of URLs.**
A queue gives O(1) enqueue/dequeue and processes each URL exactly once in FIFO order, so total
work scales linearly with the number of discovered URLs (`O(n)`). Repeatedly scanning the whole
URL collection to find "not yet crawled" entries costs `O(n)` *per lookup*, making the overall
process `O(n²)` as the collection grows — and it doesn't naturally support multiple concurrent
workers pulling distinct work items the way a queue does.

**6. Analyze the performance of the crawler under different numbers of concurrent requests
using JMeter.**
Run the JMeter plan at increasing thread counts (10 → 50 → 100 → 200) and record throughput,
average/percentile latency, and error rate in a table like:

| Concurrent Users | Avg Response Time (ms) | Throughput (req/s) | Error % |
|---|---|---|---|
| 10  | *(fill in)* | *(fill in)* | *(fill in)* |
| 50  | *(fill in)* | *(fill in)* | *(fill in)* |
| 100 | *(fill in)* | *(fill in)* | *(fill in)* |
| 200 | *(fill in)* | *(fill in)* | *(fill in)* |

Expected trend: response time for `/start` stays low (it just enqueues and returns 202) since
crawling itself happens asynchronously; the bottleneck instead shows up in the fixed-size
crawler thread pool (`crawler.thread-pool-size`) — beyond that many simultaneous crawl jobs,
work queues up in Redis rather than the HTTP layer, and `/status` throughput/latency should
remain stable even as `/start` traffic increases, since Redis operations (`INCR`, `SADD`,
`LPUSH/RPOP`) stay O(1) regardless of load.

---

## 9. Notes / Possible Extensions

- Add `robots.txt` compliance checking before enqueueing a URL.
- Add a `politeness delay` (rate-limit per domain) to avoid overloading target sites.
- Persist `PageCrawlResult` history to a relational DB for long-term analytics, using Redis
  purely as the hot-path cache.
- Add Spring Actuator + Micrometer for richer metrics during JMeter runs.
