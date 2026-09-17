# Web Crawler Service

A scalable, Redis-backed Web Crawler built with **Spring Boot 3**, **Jsoup**, and **Redis**. It provides a RESTful API to initiate web crawling from a seed URL, tracking visited URLs, discovered links, and queue status in real time.

---

## 🏗 System Architecture

The crawler uses Redis data structures for high-performance crawling tracking:
- **Redis Queue (`crawler:{id}:queue`)**: List (FIFO) storing URLs to be processed.
- **Redis Visited Set (`crawler:{id}:visited`)**: Set preventing duplicate crawling of visited pages.
- **Redis Discovered Set (`crawler:{id}:discovered`)**: Set tracking all unique URLs found across pages.
- **Redis Status Hash (`crawler:{id}:status`)**: Hash storing live status (`RUNNING`, `COMPLETED`), seed URL, last visited page, and error messages.

---

## 📋 Prerequisites

- **Docker & Docker Compose** (for running Redis & containerized application)

---

## 🚀 Setup & Execution (Docker Compose)

This mode launches both the **Redis container** and the **Web Crawler application container** simultaneously using multi-stage Docker builds.

1. Navigate to the `web-crawler` project directory:
   ```bash
   cd web-crawler
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

3. The application will start on port `8080`.

---

## 🧪 Running Test Cases

The project contains unit and integration test suites for `CrawlerService`, `CrawlerController`, `CrawlRequest`, and overall Spring context initialization.

To run all automated test cases, execute:

- **On Linux/macOS**:
  ```bash
  ./mvnw test
  ```

- **On Windows**:
  ```cmd
  .\mvnw.cmd test
  ```

### Test Summary

| Test Class | Target Component | Description |
|---|---|---|
| `CrawlerServiceTest` | `CrawlerService` | Validates URL scheme verification, malformed URL handling, and Redis queue/set operations. |
| `CrawlerControllerTest` | `CrawlerController` | MockMvc integration test verifying `POST /api/crawl` endpoint request/response formatting. |
| `CrawlRequestTest` | `CrawlRequest` | Tests DTO default values and property accessors. |
| `WebCrawlerApplicationTests` | `WebCrawlerApplication` | Verifies full Spring Boot application context loads cleanly. |

### Sample Test Output Console Window

When running `.\mvnw.cmd test`, the console displays detailed telemetry for each test run:

```text
========================================================
         CONTROLLER API TEST - REQUEST & RESPONSE       
========================================================
 POST /api/crawl Payload:
{
    "seedUrl": "https://example.com",
    "maxPages": 5
}
 Response Telemetry:
  - Seed URL               : https://example.com
  - Total Crawl Limit      : 5
  - Number of Crawls       : 2
  - Number of Websites Found: 5
  - Duplicate Websites Found: 1
  - URLs Crawled List      : [https://example.com, https://example.com/about]
========================================================

========================================================
              CRAWLER TEST RESULT SUMMARY               
========================================================
 Crawl ID               : c7b2d1e0-6ffc-4939-97ca-07b215b8c69d
 Seed URL               : https://example.com
 Status                 : COMPLETED
 Total Crawl Limit      : 5
 Number of Crawls       : 0
 Number of Websites Found: 1
 Duplicate Websites Found: 0
 URLs Crawled List      : [https://example.com]
========================================================

[INFO] Results:
[INFO] Tests run: 6, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

---

## 📡 API Reference & Usage

### Start Crawling Job

- **Endpoint**: `POST /api/crawl`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "seedUrl": "https://example.com",
    "maxPages": 5
  }
  ```

#### Example cURL Command

```bash
curl -X POST http://localhost:8080/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "seedUrl": "https://example.com",
    "maxPages": 5
  }'
```

#### Example Response Body

```json
{
  "crawlId": "d3b07384-d113-463d-a77b-bf77a06c57d7",
  "seedUrl": "https://example.com",
  "status": "COMPLETED",
  "totalCrawlLimit": 5,
  "numberOfCrawls": 5,
  "numberOfWebsitesFound": 12,
  "duplicateWebsitesFound": 3,
  "urlsCrawled": [
    "https://example.com",
    "https://example.com/about",
    "https://example.com/contact",
    "https://example.com/services",
    "https://example.com/faq"
  ],
  "processedPages": 5,
  "discoveredUrls": 12,
  "visitedUrls": 5,
  "remainingQueue": 7
}
```

---

## 📮 Manual Testing via Postman

Follow these step-by-step instructions to test the web crawler API manually using Postman:

1. **Open Postman** and create a new request tab by clicking the **+** icon.
2. **Set Request Method**:
   - Select **`POST`** from the HTTP method dropdown menu.
3. **Enter Target URL**:
   - `http://localhost:8080/api/crawl`
4. **Configure Request Headers**:
   - Click on the **Headers** tab.
   - Add Key: `Content-Type` | Value: `application/json`
5. **Configure Request Body**:
   - Click on the **Body** tab.
   - Select the **raw** radio button.
   - Set the format type dropdown to **JSON**.
   - Paste the following payload:
     ```json
     {
       "seedUrl": "https://example.com",
       "maxPages": 5
     }
     ```
6. **Send Request**:
   - Click the **Send** button.
7. **Verify Response**:
   - Response Status: `200 OK`
   - Response Body format:
     ```json
     {
       "crawlId": "d3b07384-d113-463d-a77b-bf77a06c57d7",
       "seedUrl": "https://example.com",
       "status": "COMPLETED",
       "totalCrawlLimit": 5,
       "numberOfCrawls": 5,
       "numberOfWebsitesFound": 12,
       "duplicateWebsitesFound": 3,
       "urlsCrawled": [
         "https://example.com",
         "https://example.com/about",
         "https://example.com/contact",
         "https://example.com/services",
         "https://example.com/faq"
       ],
       "processedPages": 5,
       "discoveredUrls": 12,
       "visitedUrls": 5,
       "remainingQueue": 7
     }
     ```

---

## 🛠 Resolved Issues & Fixes Applied

1. **Test Failure Fix**: Added missing `spring-boot-starter-test` dependency in `pom.xml`.
2. **Java 17 Compatibility**: Removed outdated `<source>7</source>` / `<target>7</target>` configuration in `maven-compiler-plugin` to align with Spring Boot 3 & JDK 17.
3. **Multi-Stage Docker Build**: Updated `Dockerfile` to compile and package the app inside Docker autonomously, eliminating pre-build dependencies.
4. **Flexible Redis Host Configuration**: Configured `spring.data.redis.host=${SPRING_DATA_REDIS_HOST:localhost}` in `application.properties` so the crawler seamlessly works both locally and inside Docker networks.
5. **Validation Handling**: Preserved specific exception messages in `CrawlerService.validateUrl`.
