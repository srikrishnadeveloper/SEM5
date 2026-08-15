# URL Shortener + Redis Caching — UCS3513 System Design Lab, Ex. 3

Same Spring Boot app as Experiment 2, but `GET /{shortCode}` now checks Redis
before it checks the database (cache-aside pattern).

## Files

- `UrlShortenerApplication.java` — starts the app
- `UrlMapping.java` — the database table (id, shortCode, longUrl)
- `UrlMappingRepository.java` — talks to the database
- `UrlShortenerController.java` — the two REST APIs + hashing logic + redis caching
  (has two big comment blocks at the bottom of the file: what redis actually
  is, and how the 20MB cap / allkeys-lru eviction works — read those if you
  forgot)
- `ShortenRequest.java` / `ShortenResponse.java` — JSON request/response
- `application.properties` — database + redis config (does NOT include the
  20MB cap/eviction policy — that's set on the redis server itself, see below)

## What's different from Experiment 2

Only 3 files actually changed: `pom.xml` (added the redis dependency),
`application.properties` (added redis host/port), and
`UrlShortenerController.java` (the `GET /{shortCode}` method now checks redis
first). Everything else is identical to Experiment 2.

On top of that, two more things were added after the initial caching version:

- **20MB storage cap + allkeys-lru eviction** — set directly on the
  redis/memurai server (not in any Java/properties file), so the cache can
  never grow past 20MB; once full it evicts whichever key was least recently
  used. See "Storage cap + eviction policy" below.
- **MySQL migration — requested but not done yet.** Still running on H2.
  No MySQL server is installed on this machine, so `pom.xml` and
  `application.properties` were deliberately left untouched rather than
  half-wiring a connection that can't actually run. Nothing in the code
  assumes H2 specifically (Spring Data JPA + Hibernate would work the same
  way against MySQL), so switching later is just: swap the H2 dependency for
  `mysql-connector-j` in `pom.xml`, and change the 4
  `spring.datasource.*` lines in `application.properties` to point at a real
  MySQL instance.

## Prerequisites

Unlike Experiment 2 (which only needed Java + Maven, since H2 is in-memory and
built in), this one needs an actual Redis server running and reachable at
`localhost:6379` — the app will fail to start without it.

**On this laptop:** we already have Memurai (a Windows build of Redis)
installed and running as a background service, so nothing extra is needed
here.

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
mentioning port 6379, redis isn't running yet — see Prerequisites above and
start it first, then re-run this command.

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

The first call is a cache miss (goes to the H2 database, slower). The second
call is a cache hit (comes straight from redis, noticeably faster) — that's
the whole point of this experiment.

**Check what's actually sitting in the cache** (optional, but good for
proving it's real and not just a fast database):
```bash
redis-cli GET aB12Xy
```
(`memurai-cli.exe GET aB12Xy` on Windows if redis-cli isn't on PATH)

You can also test all of this in Postman, exactly as the lab sheet asks.
