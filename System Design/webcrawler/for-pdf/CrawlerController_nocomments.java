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

@RestController
@RequestMapping("/api/crawler")
public class CrawlerController {

    private final CrawlerService crawlerService;

    private final RedisCrawlStore store;

    public CrawlerController(CrawlerService crawlerService, RedisCrawlStore store) {
        this.crawlerService = crawlerService;
        this.store = store;
    }

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

    @GetMapping("/status")
    public ResponseEntity<CrawlJobStatusResponse> status() {
        CrawlJobState state;
        if (crawlerService.isJobRunning()) {
            state = CrawlJobState.RUNNING;
        } else if (crawlerService.getCurrentSeedUrl() == null) {
            state = CrawlJobState.PENDING;
        } else {
            state = CrawlJobState.COMPLETED;
        }

        CrawlJobStatusResponse resp = new CrawlJobStatusResponse(
                "current",
                crawlerService.getCurrentSeedUrl(),
                state,
                store.queueSize(),
                store.visitedCount(),
                crawlerService.isJobRunning() ? -1 : getEffectiveCrawledCount(),
                store.getFailedCount(),
                crawlerService.getJobStartedAt(),
                Instant.now()
        );
        return ResponseEntity.ok(resp);
    }

    @GetMapping("/visited")
    public ResponseEntity<Set<String>> visited() {
        return ResponseEntity.ok(store.getVisited());
    }

    @GetMapping("/queue")
    public ResponseEntity<?> queue(@RequestParam(defaultValue = "50") long limit) {
        List<String> items = store.peekQueue(limit);
        return ResponseEntity.ok(Map.of("queueSize", store.queueSize(), "sample", items));
    }

    @GetMapping("/results")
    public ResponseEntity<List<PageCrawlResult>> results() {
        return ResponseEntity.ok(store.getAllPageResults());
    }

    @GetMapping("/results/page")
    public ResponseEntity<?> resultForUrl(@RequestParam String url) {
        PageCrawlResult result = store.getPageResult(url);
        if (result == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("message", "No crawl result found for " + url));
        }
        return ResponseEntity.ok(result);
    }

    @PostMapping("/stop")
    public ResponseEntity<?> stop() {
        crawlerService.stop();
        return ResponseEntity.ok(Map.of("message", "Crawl stopped"));
    }

    @DeleteMapping("/reset")
    public ResponseEntity<?> reset() {
        crawlerService.stop();
        store.resetAll();
        return ResponseEntity.ok(Map.of("message", "Crawler state reset"));
    }

    @GetMapping("/health")
    public ResponseEntity<?> health() {
        return ResponseEntity.ok(Map.of("status", "UP", "timestamp", Instant.now().toString()));
    }

    private long getEffectiveCrawledCount() {
        long count = store.getCrawledCount();
        return count >= 0 ? count : 0;
    }
}
