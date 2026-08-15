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
