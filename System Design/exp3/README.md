# URL Shortener + Redis Caching — UCS3513 System Design Lab, Ex. 3

Same Spring Boot app as Experiment 2, but `GET /{shortCode}` now checks Redis
before it checks the database (cache-aside pattern).

## Files

- `UrlShortenerApplication.java` — starts the app
- `UrlMapping.java` — the document stored in MongoDB (id, shortCode, longUrl)
- `UrlMappingRepository.java` — talks to MongoDB
- `UrlShortenerController.java` — the two REST APIs + hashing logic + redis caching
  (has two big comment blocks at the bottom of the file: what redis actually
  is, and how the 20MB cap / allkeys-lru eviction works — read those if you
  forgot)
- `ShortenRequest.java` / `ShortenResponse.java` — JSON request/response
- `application.properties` — MongoDB + redis config (does NOT include the
  20MB cap/eviction policy — that's set on the redis server itself, see below)

## What's different from Experiment 2

Started as just 3 changed files (`pom.xml`, `application.properties`,
`UrlShortenerController.java` — added redis caching). Two more things were
added after that:

- **20MB storage cap + allkeys-lru eviction** — set directly on the
  redis/memurai server (not in any Java/properties file), so the cache can
  never grow past 20MB; once full it evicts whichever key was least recently
  used. See "Storage cap + eviction policy" below.
- **Database swapped from H2 to MongoDB.** MySQL was asked for first, but no
  MySQL server was available on this machine (install kept failing on a
  package fetch + this shell not being elevated), so we went with MongoDB
  instead since it was already installed and running locally. This changed
  4 files: `pom.xml` (h2 + jpa dependencies swapped for
  `spring-boot-starter-data-mongodb`), `application.properties` (jdbc/h2/jpa
  lines replaced with `spring.data.mongodb.*`), `UrlMapping.java`
  (`@Entity`/`@GeneratedValue` → `@Document`, id type `Long` → `String` since
  mongo ids are ObjectId strings not auto-increment numbers), and
  `UrlMappingRepository.java` (`JpaRepository<UrlMapping, Long>` →
  `MongoRepository<UrlMapping, String>`). `UrlShortenerController.java` did
  **not** need any changes for this — it never touched `getId()` or the id
  type directly, only `findByShortCode()`, which works the same either way.

## Prerequisites

Unlike Experiment 2 (which only needed Java + Maven, since H2 is in-memory and
built in), this one needs two actual servers running: **Redis**
(`localhost:6379`) and **MongoDB** (`localhost:27017`) — the app will fail to
start without both.

**On this laptop:** Memurai (Redis for Windows) and MongoDB Server are both
already installed and running as background services, so nothing extra is
needed here.

**MongoDB on a friend's laptop / any machine that doesn't have it:** install
MongoDB Community Server from mongodb.com/try/download/community, or run it
in Docker instead: `docker run --name mongo -p 27017:27017 -d mongo`. Either
way it needs to be reachable at `localhost:27017` before starting the app.

**On a friend's laptop / any machine that doesn't have this already:**
you need to get *something* Redis-compatible running on port 6379 first.
Pick whichever is easiest for that machine:

- **Docker (works the same on Windows/Mac/Linux, easiest if Docker is already installed):**
  ```bash
  docker run --name redis -p 6379:6379 -d redis
  ```
  This downloads and starts real Redis in a container. To start it again after
  a reboot: `docker start redis`.

- **Windows, no Docker — install Memurai (what we used):**
  Download the free "Memurai Developer" edition from memurai.com and install
  it. It installs as a Windows service and starts automatically — same as on
  this laptop, nothing to run manually afterwards.

- **Mac:**
  ```bash
  brew install redis
  brew services start redis
  ```

- **Linux:**
  ```bash
  sudo apt install redis-server
  sudo systemctl start redis-server
  ```

However you install it, check it's actually listening before running the app:
```bash
redis-cli ping
```
(or `memurai-cli ping` on Windows) should reply `PONG`. If that command isn't
found, redis/memurai isn't running yet.

### Storage cap + eviction policy (20MB, allkeys-lru)

Redis is capped at 20MB total and set to evict the least-recently-used key
first once it's full — this is set on the redis/memurai server itself, not
in the Spring app, so it needs to be set again on any new machine:

```bash
redis-cli CONFIG SET maxmemory 20mb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG REWRITE
```
(`memurai-cli.exe` instead of `redis-cli` on Windows). `CONFIG REWRITE` saves
it into the config file so it survives a restart — without it, the cap resets
back to default the next time redis/memurai restarts.

## Run it

```bash
cd exp3
mvn spring-boot:run
```

Runs on `http://localhost:8080`. If it fails to start with a connection error
mentioning port 6379, redis isn't running yet; port 27017 means mongodb isn't
running yet — see Prerequisites above and start whichever one first, then
re-run this command.

## Test it

**Shorten a URL:**
```bash
curl -X POST http://localhost:8080/shorten \
  -H "Content-Type: application/json" \
  -d "{\"longUrl\": \"https://www.google.com/search?q=system+design\"}"
```

Response:
```json
{ "shortUrl": "http://localhost:8080/aB12Xy" }
```

**Visit the short URL twice** and watch the timing difference:
```bash
curl -w "\ntime_total: %{time_total}s\n" -L http://localhost:8080/aB12Xy
curl -w "\ntime_total: %{time_total}s\n" -L http://localhost:8080/aB12Xy
```

The first call is a cache miss (goes to MongoDB, slower). The second call is
a cache hit (comes straight from redis, noticeably faster) — that's the
whole point of this experiment.

**Check what's actually sitting in the cache** (optional, but good for
proving it's real and not just a fast database):
```bash
redis-cli GET aB12Xy
```
(`memurai-cli.exe GET aB12Xy` on Windows if redis-cli isn't on PATH)

**Check what's actually sitting in MongoDB** (same idea, for the database
side):
```bash
mongosh --eval 'db.getSiblingDB("urlshortenerdb").url_mapping.find()'
```

You can also test all of this in Postman, exactly as the lab sheet asks.

## Benchmark it

Same idea as exp2's benchmark script, but adapted for caching: it runs
`POST /shorten` 100 times, then `GET /{shortCode}` 100 times **on the same
shortcode**, since that's what actually demonstrates caching — request 1 is
a cache miss (Mongo), requests 2-100 are cache hits (Redis).

```powershell
.\benchmark.ps1
```
(or `bash benchmark.sh` from Git Bash)

Make sure the app, MongoDB, and Redis/Memurai are all running first. It
prints:
- the average for `POST /shorten`
- the GET average split into cache-miss time vs. cache-hit average, plus the
  speedup multiplier between them
- the **real cache hit rate / miss rate**, read directly from Redis's own
  `INFO stats` counters (`keyspace_hits` / `keyspace_misses`) rather than
  just assumed from the request pattern — so it double-checks itself

Example output:
```
POST /shorten            average: 14.42 ms  (100/100 requests succeeded)
GET  /{shortCode} overall  average: 11.41 ms  (100/100 requests succeeded)
  - request 1 (cache MISS, from mongo):  889.61 ms
  - requests 2-100 (cache HITS, from redis) average: 2.54 ms
  - cache made it about 350.4x faster

cache hit rate:  99.0%  (99 hits out of 100 redis lookups)
cache miss rate: 1.0%  (1 misses out of 100 redis lookups)
```

Since Redis is capped at 20MB with `allkeys-lru` eviction (see "Storage cap +
eviction policy" above), running this benchmark repeatedly without ever
restarting Redis will eventually start evicting older benchmark keys once
the cache fills up — that's expected behaviour, not a bug, and is exactly
what the eviction policy is there to demonstrate.
