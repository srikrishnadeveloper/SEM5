# URL Shortener (Basic) — UCS3513 System Design Lab, Ex. 2

A simple Spring Boot app that shortens URLs using SHA-256 hashing.

## Files

- `UrlShortenerApplication.java` — starts the app
- `UrlMapping.java` — the database table (id, shortCode, longUrl)
- `UrlMappingRepository.java` — talks to the database
- `UrlShortenerController.java` — the two REST APIs + hashing logic
- `ShortenRequest.java` / `ShortenResponse.java` — JSON request/response
- `application.properties` — database config

## How the short code works

1. Hash the long URL with SHA-256 → get a hex string.
2. Take the first 6 characters as the short code.
3. Check the database — if that code is already used, take 7 characters,
   then 8, until it's free.
4. Save the (shortCode, longUrl) pair to the database.

## Run it

```bash
cd url-shortener
mvn spring-boot:run
```

Runs on `http://localhost:8080` with an in-memory H2 database — no setup needed.

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

**Visit the short URL** (paste in a browser, or use `curl -L`):
```bash
curl -L http://localhost:8080/aB12Xy
```

It redirects to the original long URL.

You can also test both steps in Postman, exactly as the lab sheet asks.

## Benchmark it

There's a script that runs each endpoint 100 times and prints the average
response time — handy for comparing against exp3's cached version.

```powershell
.\benchmark.ps1
```
(or `bash benchmark.sh` from Git Bash)

Make sure the app is running first (`mvn spring-boot:run`). It prints the
average for `POST /shorten` and `GET /{shortCode}` separately, plus how many
of the 100 requests actually succeeded.

Note: since exp2 has no caching, every `GET` hits the H2 database fresh every
time — this is the "before caching" baseline that exp3's Redis layer is
built to beat. If you run this a lot without restarting the app, occasionally
a `POST` may fail with a `NonUniqueResultException` server-side — that's a
real quirk in this exp's code (`shortCode` has no unique DB constraint, and
the uniqueness-check loop gives up after 8 characters), not a benchmark
script bug. Restarting the app clears it since H2 is in-memory.
