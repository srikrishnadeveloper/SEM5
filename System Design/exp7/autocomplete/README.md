# Autocomplete Search System

Simple Spring Boot + Redis autocomplete with Trie prefix search.

## Files

- `pom.xml` — Maven build file
- `src/main/resources/application.properties` — Redis config
- `src/main/resources/terms.csv` — search terms and frequencies (dataset)
- `src/main/java/com/ssn/autocomplete/`
  - `AutocompleteApplication.java` — main class
  - `AutocompleteService.java` — Trie + Redis cache logic
  - `AutocompleteController.java` — REST API
- `Dockerfile` — Docker image build
- `docker-compose.yml` — app + Redis together
- `jmeter-test.jmx` — JMeter load test

## Prerequisites

- Java 17 or higher
- Maven
- Redis running on `localhost:6379`
  - Option A: Install Memurai / Redis on Windows
  - Option B: Start Docker Desktop and run `docker-compose up --build`
- Docker Desktop (if using Docker)
- JMeter (if running load test)

## Run locally (without Docker)

1. Make sure Redis is running on `localhost:6379`.
2. Build the project:
   ```bash
   mvn clean package -DskipTests
   ```
3. Run the app:
   ```bash
   mvn spring-boot:run
   ```
4. Test it in another terminal:
   ```bash
   curl http://localhost:8080/api/health
   curl "http://localhost:8080/api/search?prefix=app&k=4"
   ```

Expected output for the search:
```json
["apple (freq=1200)","application (freq=950)","appointment (freq=800)","app store (freq=600)"]
```

## Run with Docker

1. Open Docker Desktop and wait until it says "Docker Desktop is running".
2. Build and start both containers:
   ```bash
   docker-compose up --build
   ```
3. Test:
   ```bash
   curl http://localhost:8080/api/health
   curl "http://localhost:8080/api/search?prefix=app&k=4"
   ```

## Run JMeter load test

1. Open Apache JMeter.
2. File → Open → select `jmeter-test.jmx`.
3. Click the green Start button.
4. View results in `View Results Tree` or `Summary Report`.

The test plan sends 10 users × 100 loops = 1000 requests to `/api/search?prefix=app&k=4`.

## How it works

1. At startup, the app reads `terms.csv` and builds a Trie.
2. A request to `/api/search?prefix=...&k=...` first checks Redis.
3. If the prefix is cached, the result is returned immediately.
4. If not, the Trie is searched, top-K terms are sorted by frequency, stored in Redis for 5 minutes, and returned.

## Changing the dataset

Edit `src/main/resources/terms.csv` and restart the app. Each line must be:
```
term,frequency
```

Example:
```
apple,1200
banana,700
cat,900
```
