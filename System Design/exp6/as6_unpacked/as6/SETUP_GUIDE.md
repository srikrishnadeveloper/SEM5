# Step-by-Step Guide: API Rate Limiter (Token Bucket)

Follow these steps in order to get the project running and tested from scratch.

---

## Step 1: Prerequisites

Make sure you have installed:
- **Java 17 or higher** — check with `java -version`
- **Maven** — check with `mvn -version`
- **Docker** — check with `docker -version`

| Tool | Linux/Mac check | Windows check |
|------|----------------|---------------|
| Java | `java -version` | `java -version` (in CMD/PowerShell) |
| Maven | `mvn -version` | `mvn -version` (needs PATH — see note below) |
| Docker | `docker -version` | `docker -version` (Docker Desktop must be running) |

> **Windows note on Maven:** If `mvn` is not recognized, Maven may not be on your PATH. If installed via NetBeans, add `C:\Program Files\Apache NetBeans\java\maven\bin` to your system PATH (Settings > System > About > Advanced system settings > Environment Variables > Path > Edit > New).

If any of these are missing, install them before continuing.

---

## Step 2: Extract the project

Unzip `rate-limiter.zip` anywhere on your machine, then move into the folder:

```bash
cd rate-limiter
```

You should see:
```
pom.xml
README.md
test-rate-limiter.sh
test-rate-limiter.bat
src/main/java/com/ssn/ratelimiter/
src/main/resources/application.properties
```

---

## Step 3: Start Redis using Docker

**Linux / Mac / WSL:**
```bash
docker run -d --name redis-ratelimiter -p 6379:6379 redis:7
```

**Windows (CMD / PowerShell):**
```cmd
docker run -d --name redis-ratelimiter -p 6379:6379 redis:7
```
> Make sure Docker Desktop is running (check the system tray icon — it should say "Docker Desktop is running").

Verify it's running:
```bash
docker ps
```
You should see a container named `redis-ratelimiter` with status `Up`.

> If you already ran this before and the container exists but is stopped, use `docker start redis-ratelimiter` instead.

---

## Step 4: Build and run the Spring Boot app

From inside the `rate-limiter` folder:

```bash
mvn spring-boot:run
```

Wait for the console to show Spring Boot has started (look for a line like `Started RateLimiterApplication in ... seconds`). Leave this terminal running — this is your server.

---

## Step 5: Confirm the app is up

Open a **new** terminal window (keep the server running in the first one) and run:

**Linux / Mac / WSL:**
```bash
curl -i http://localhost:8080/api/health
```

**Windows (CMD / PowerShell):**
```cmd
curl.exe -i http://localhost:8080/api/health
```

Expected output: `HTTP/1.1 200 OK` with body `Service is healthy`.

---

## Step 6: Test a single normal request

**Linux / Mac / WSL:**
```bash
curl -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Windows (CMD / PowerShell):**
```cmd
curl.exe -i http://localhost:8080/api/resource -H "client-id: user1"
```

Expected: `200 OK — Request processed successfully`.

---

## Step 7: Check remaining tokens

**Linux / Mac / WSL:**
```bash
curl -i http://localhost:8080/api/rate-limit-status -H "client-id: user1"
```

**Windows (CMD / PowerShell):**
```cmd
curl.exe -i http://localhost:8080/api/rate-limit-status -H "client-id: user1"
```

Expected: something like `Remaining tokens: 9` (started at 10, one was just used).

---

## Step 8: Run the burst test script

**Linux / Mac / WSL:**
```bash
chmod +x test-rate-limiter.sh
./test-rate-limiter.sh
```

**Windows (CMD):**
```cmd
test-rate-limiter.bat
```
> Both scripts accept optional arguments: number of requests and client-id.
> e.g. `test-rate-limiter.bat 20 user2` or `./test-rate-limiter.sh 20 user2`

Expected: the first ~10 requests print `200 OK`, the remaining ones print `429 TOO MANY REQUESTS`, followed by a summary and the current token count.

---

## Step 9: Watch tokens refill

Wait a few seconds, then try again:

**Linux / Mac / WSL:**
```bash
sleep 5
curl -i http://localhost:8080/api/resource -H "client-id: user1"
```

**Windows (CMD):**
```cmd
timeout /t 5 /nobreak >nul
curl.exe -i http://localhost:8080/api/resource -H "client-id: user1"
```

Expected: `200 OK` again, since tokens refill at 1 per second by default.

---

## Step 10: Confirm independent buckets per client

**Linux / Mac / WSL:**
```bash
curl -i http://localhost:8080/api/resource -H "client-id: user2"
```

**Windows (CMD / PowerShell):**
```cmd
curl.exe -i http://localhost:8080/api/resource -H "client-id: user2"
```

Expected: `200 OK` even if `user1` is currently rate-limited, since each `client-id` has its own bucket.

---

## Step 11 (optional): Inspect Redis directly

```bash
docker exec -it redis-ratelimiter redis-cli
keys rate_limit:*
get rate_limit:user1:tokens
get rate_limit:user1:last_refill
exit
```

This shows the faculty the actual state Redis is storing — proof the rate limit is backed by Redis, not just an in-memory variable.

---

## Step 12: Shut down when done

In the terminal running the Spring Boot app, press `Ctrl+C` to stop it.

Stop (and optionally remove) the Redis container:
```bash
docker stop redis-ratelimiter
docker rm redis-ratelimiter   # only if you want to delete it entirely
```

---

## Quick troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect` | Spring Boot app isn't running yet | Check the terminal from Step 4 for errors, make sure it fully started |
| App fails to start with a Redis connection error | Redis container isn't running | Run `docker ps`; if missing, redo Step 3 |
| Every request returns 429 immediately | Bucket wasn't full to begin with (tested earlier in the same session) | Wait a bit for refill, or use a new `client-id` |
| `mvn: command not found` | Maven not installed / not on PATH | Install Maven and retry `mvn -version` |
| `curl` in PowerShell calls `Invoke-WebRequest` | PowerShell aliases `curl` to its own cmdlet | Use `curl.exe` instead of `curl` |
| `sleep` not recognized (Windows CMD) | `sleep` is a Linux command | Use `timeout /t 5 /nobreak >nul` instead |
| Docker command fails on Windows | Docker Desktop not running | Start Docker Desktop from Start menu, wait for it to initialize |
