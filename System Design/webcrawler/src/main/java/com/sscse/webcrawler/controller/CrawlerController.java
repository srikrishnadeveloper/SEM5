package com.sscse.webcrawler.controller;

import com.sscse.webcrawler.model.*;
import com.sscse.webcrawler.service.CrawlerService;
import com.sscse.webcrawler.service.RedisCrawlStore;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * REST API surface for the web crawler.
 * Test these endpoints with Postman or the JMeter test plan included under /jmeter.
 * 
 * All endpoints are under /api/crawler.
 */
@RestController
@RequestMapping("/api/crawler")
public class CrawlerController {

    /** The core crawling engine that does the actual work. */
    private final CrawlerService crawlerService;

    /** Thin Redis wrapper for queue, visited set, page data, and stats. */
    private final RedisCrawlStore store;

    /**
     * Constructor-based dependency injection.
     * @param crawlerService The crawler engine service
     * @param store Redis wrapper for state persistence
     */
    public CrawlerController(CrawlerService crawlerService, RedisCrawlStore store) {
        this.crawlerService = crawlerService;
        this.store = store;
    }

    /** POST /api/crawler/start -- kick off a new crawl job (async, returns immediately). */
    @PostMapping("/start")
    public ResponseEntity<?> start(@Valid @RequestBody CrawlRequest request) {
        crawlerService.startCrawl(request);
        return ResponseEntity.accepted().body(Map.of(
                "message", "Crawl started",
                "seedUrl", request.getSeedUrl(),
                "maxDepth", request.getMaxDepth(),
                "maxPages", request.getMaxPages()
        ));
    }

    /** GET /api/crawler/status -- current job + queue/visited/crawled/failed counts. */
    @GetMapping("/status")
    public ResponseEntity<CrawlJobStatusResponse> status() {
        // Determine current state: running, pending, or completed/stopped
        CrawlJobState state;
        if (crawlerService.isJobRunning()) {
            state = CrawlJobState.RUNNING;
        } else if (crawlerService.getCurrentSeedUrl() == null) {
            state = CrawlJobState.PENDING;
        } else {
            state = CrawlJobState.COMPLETED;
        }

        // Build the status response with current state and statistics
        CrawlJobStatusResponse resp = new CrawlJobStatusResponse(
                "current",  // jobId
                crawlerService.getCurrentSeedUrl(),
                state,
                store.queueSize(),                    // queuedCount
                store.visitedCount(),                 // visitedCount
                crawlerService.isJobRunning() ? -1 : getEffectiveCrawledCount(), // crawledCount
                store.getFailedCount(),               // failedCount
                crawlerService.getJobStartedAt(),     // startedAt
                Instant.now()                         // updatedAt
        );
        return ResponseEntity.ok(resp);
    }

    /** GET /api/crawler/visited -- the visited-URL set (dedupe record). */
    @GetMapping("/visited")
    public ResponseEntity<Set<String>> visited() {
        return ResponseEntity.ok(store.getVisited());
    }

    /** GET /api/crawler/queue?limit=50 -- peek at URLs currently waiting to be crawled. */
    @GetMapping("/queue")
    public ResponseEntity<?> queue(@RequestParam(defaultValue = "50") long limit) {
        List<String> items = store.peekQueue(limit);
        return ResponseEntity.ok(Map.of("queueSize", store.queueSize(), "sample", items));
    }

    /** GET /api/crawler/results -- full crawl metadata (title, links, status) for every page crawled. */
    @GetMapping("/results")
    public ResponseEntity<List<PageCrawlResult>> results() {
        return ResponseEntity.ok(store.getAllPageResults());
    }

    /** GET /api/crawler/results/page?url=... -- metadata for one specific URL. */
    @GetMapping("/results/page")
    public ResponseEntity<?> resultForUrl(@RequestParam String url) {
        PageCrawlResult result = store.getPageResult(url);
        if (result == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("message", "No crawl result found for " + url));
        }
        return ResponseEntity.ok(result);
    }

    /** POST /api/crawler/stop -- stop the current job and drain the queue. */
    @PostMapping("/stop")
    public ResponseEntity<?> stop() {
        crawlerService.stop();
        return ResponseEntity.ok(Map.of("message", "Crawl stopped"));
    }

    /** DELETE /api/crawler/reset -- clear queue, visited set, results, and stats (fresh start). */
    @DeleteMapping("/reset")
    public ResponseEntity<?> reset() {
        crawlerService.stop();
        store.resetAll();
        return ResponseEntity.ok(Map.of("message", "Crawler state reset"));
    }

    /** GET /api/crawler/health -- simple liveness probe (also useful as a JMeter warm-up target). */
    @GetMapping("/health")
    public ResponseEntity<?> health() {
        return ResponseEntity.ok(Map.of("status", "UP", "timestamp", Instant.now().toString()));
    }

    /** Helper method to get effective crawled count. */
    private long getEffectiveCrawledCount() {
        long count = store.getCrawledCount();
        return count >= 0 ? count : 0;
    }
}