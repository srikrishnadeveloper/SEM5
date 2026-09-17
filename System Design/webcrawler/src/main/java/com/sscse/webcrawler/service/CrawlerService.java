package com.sscse.webcrawler.service;

import com.sscse.webcrawler.model.CrawlRequest;
import com.sscse.webcrawler.model.CrawlStatus;
import com.sscse.webcrawler.model.PageCrawlResult;
import com.sscse.webcrawler.util.UrlValidator;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Core crawling engine.
 * 
 * This class implements a breadth-first, queue-based crawling algorithm.
 * The crawl is asynchronous - calling startCrawl() returns immediately,
 * and crawling happens on background worker threads.
 * 
 * Algorithm steps:
 *   1. Validate the seed URL and push it onto the Redis-backed QUEUE.
 *   2. Worker threads repeatedly DEQUEUE a URL entry.
 *   3. If the URL is already in the VISITED set -> skip (avoids duplicate work).
 *   4. Otherwise: fetch the page (HTTP GET), mark it VISITED, extract <a href> links,
 *      validate + normalize each discovered link, and ENQUEUE the ones that are new
 *      and within depth/page budget.
 *   5. Repeat until the queue is empty or maxPages/maxDepth budget is exhausted.
 * 
 * Key design decisions:
 * - Depth is encoded into each queue entry as "depth|url" to survive across dequeues.
 * - Redis is used instead of pure in-memory data structures so the crawl state
 *   survives application restarts and can be shared across multiple instances.
 * - A thread pool with up to 4 workers processes URLs concurrently (relevant for
 *   JMeter load testing analysis).
 * - AtomicBoolean tracks whether a job is currently running for status checks.
 */
@Service
public class CrawlerService {

    private static final Logger log = LoggerFactory.getLogger(CrawlerService.class);

    /** Redis wrapper for queue, visited set, page data, and stats. */
    private final RedisCrawlStore store;

    /** Thread pool for concurrent URL processing. */
    private final ExecutorService executorService;

    /** Connect timeout in milliseconds for HTTP requests. */
    @Value("${crawler.connect-timeout-ms:5000}")
    private int connectTimeoutMs;

    /** User-Agent string sent with HTTP requests. */
    @Value("${crawler.user-agent:SSNCE-UCS3513-WebCrawler/1.0}")
    private String userAgent;

    /** Atomic flag: is a crawl job currently running? */
    private final AtomicBoolean jobRunning = new AtomicBoolean(false);

    /** The seed URL that was used to start the current crawl job. */
    private volatile String currentSeedUrl;

    /** Instant when the current crawl job started. */
    private volatile Instant jobStartedAt;

    /**
     * Constructor-based dependency injection.
     * @param store Redis wrapper for queue/visited/page-data/stats
     * @param executorService Thread pool for concurrent URL processing
     */
    public CrawlerService(RedisCrawlStore store, ExecutorService executorService) {
        this.store = store;
        this.executorService = executorService;
    }

    /**
     * Kicks off an asynchronous crawl job.
     * Returns immediately; crawling happens on worker threads.
     * 
     * Steps:
     * 1. Validate the seed URL using UrlValidator.
     * 2. Normalize the seed URL (strip fragment, trailing slash).
     * 3. Set currentSeedUrl and jobStartedAt.
     * 4. Set jobRunning flag to true.
     * 5. Encode depth (0) into the queue entry and enqueue the seed.
     * 6. Determine number of worker threads based on maxPages.
     * 7. Submit worker threads that execute crawlLoop().
     * 
     * @param request The crawl request payload (seedUrl, maxDepth, maxPages, sameDomainOnly)
     * @throws IllegalArgumentException if the seed URL is invalid
     */
    public void startCrawl(CrawlRequest request) {
        // Step 1: Validate the seed URL
        if (!UrlValidator.isValid(request.getSeedUrl())) {
            throw new IllegalArgumentException("Invalid seed URL: " + request.getSeedUrl());
        }

        // Step 2: Normalize the seed URL (strip fragment, trailing slash)
        String normalizedSeed = UrlValidator.normalize(request.getSeedUrl());

        // Step 3: Set current seed and job start time
        this.currentSeedUrl = normalizedSeed;
        this.jobStartedAt = Instant.now();

        // Step 4: Mark job as running
        this.jobRunning.set(true);

        // Step 5: Encode depth (0) into queue entry and enqueue the seed
        store.enqueue(encodeDepth(normalizedSeed, 0));

        // Step 6: Determine number of worker threads
        int maxDepth = request.getMaxDepth();
        int maxPages = request.getMaxPages();
        boolean sameDomainOnly = request.isSameDomainOnly();

        // Atomic counter to track how many pages have been processed
        AtomicInteger pagesProcessed = new AtomicInteger(0);

        // Step 7: Fan out multiple workers (1 to 4, based on maxPages)
        int workers = Math.min(4, Math.max(1, maxPages));
        for (int i = 0; i < workers; i++) {
            executorService.submit(() ->
                    crawlLoop(normalizedSeed, maxDepth, maxPages, sameDomainOnly, pagesProcessed));
        }
    }

    /**
     * Worker loop that processes URLs from the queue until the budget is exhausted.
     * 
     * Steps per iteration:
     *   1. Dequeue a URL entry. If null and queue is empty, break.
     *   2. Decode the depth from the queue entry.
     *   3. If depth exceeds maxDepth, discard and continue.
     *   4. Process the URL (check visited, fetch, extract links, enqueue new ones).
     *   5. Increment the pagesProcessed counter.
     *   6. Repeat while pagesProcessed < maxPages.
     * 
     * When the loop exits, mark the job as no longer running.
     * 
     * @param seedUrl The original seed URL for the crawl
     * @param maxDepth Maximum crawling depth budget
     * @param maxPages Maximum number of pages to crawl
     * @param sameDomainOnly If true, only follow links within the same domain
     * @param pagesProcessed Atomic counter for pages processed so far
     */
    private void crawlLoop(String seedUrl, int maxDepth, int maxPages,
                           boolean sameDomainOnly, AtomicInteger pagesProcessed) {
        while (pagesProcessed.get() < maxPages) {
            // Step 1: Dequeue the next URL entry
            String entry = store.dequeue();
            if (entry == null) {
                // Queue temporarily empty; small backoff then re-check
                // Another worker may have added more URLs.
                if (store.queueSize() == 0) break;
                continue;
            }

            // Step 2: Decode the depth and URL from the entry
            String[] decoded = decodeDepth(entry);
            int depth = Integer.parseInt(decoded[0]);
            String url = decoded[1];

            // Step 3: If depth exceeds maxDepth, discard and continue
            if (depth > maxDepth) {
                log.debug("SKIP (depth {} exceeds budget {}): {}", depth, maxDepth, url);
                continue;
            }

            // Step 4: Process the URL (fetch, extract links, enqueue new ones)
            processUrl(url, depth, seedUrl, maxDepth, sameDomainOnly);

            // Step 5: Increment the pages processed counter
            pagesProcessed.incrementAndGet();
        }

        // Mark the job as no longer running
        jobRunning.set(false);
    }

    /**
     * Processes exactly one URL: checks visited-set, fetches page if new,
     * extracts links, validates them, and enqueues newly discovered ones.
     * 
     * Steps:
     *   1. Check if this URL has already been visited. If so, skip (duplicate avoided).
     *   2. Fetch the page using Jsoup (HTTP GET with user-agent and timeout).
     *   3. Extract all valid, normalized hyperlinks from the page.
     *   3. Save the crawl result (success or failure) to Redis.
     *   4. If within depth budget, enqueue newly discovered links.
     *   5. Log the result.
     * 
     * @param url The URL to process
     * @param depth Crawling depth from seed (seed = 0)
     * @param seedUrl The original seed URL for same-domain validation
     * @param maxDepth Maximum crawling depth budget
     * @param sameDomainOnly If true, only follow links within the same domain
     */
    private void processUrl(String url, int depth, String seedUrl, int maxDepth, boolean sameDomainOnly) {
        // ---- Step: has this URL already been visited? ----
        boolean isNew = store.markVisitedIfAbsent(url);
        if (!isNew) {
            log.debug("SKIP (already visited): {}", url);
            return; // duplicate crawl avoided
        }

        try {
            // ---- Step: fetch the page ----
            Document doc = Jsoup.connect(url)
                    .userAgent(userAgent)
                    .timeout(connectTimeoutMs)
                    .get();

            // ---- Step: extract valid links ----
            List<String> discovered = extractLinks(doc, url, seedUrl, sameDomainOnly);

            // ---- Step: save the crawl result ----
            PageCrawlResult result = new PageCrawlResult(
                    url, CrawlStatus.CRAWLED, doc.title(), depth,
                    discovered.size(), discovered, Instant.now());
            store.savePageResult(result);
            store.incrementCrawled();

            log.info("CRAWLED [{}] depth={} links={}", url, depth, discovered.size());

            // ---- Step: enqueue newly discovered links (if within depth budget) ----
            if (depth < maxDepth) {
                for (String link : discovered) {
                    if (!store.isVisited(link)) {
                        store.enqueue(encodeDepth(link, depth + 1));
                    }
                }
            }

        } catch (Exception e) {
            // ---- Step: handle crawl failure ----
            PageCrawlResult failResult = new PageCrawlResult(
                    url, CrawlStatus.FAILED, null, depth, 0, List.of(),
                    e.getMessage(), Instant.now());
            store.savePageResult(failResult);
            store.incrementFailed();

            log.warn("FAILED [{}]: {}", url, e.getMessage());
        }
    }

    /**
     * Extracts, validates, and normalizes all hyperlinks found on a page.
     * 
     * Steps for each <a href> element:
     *   1. Get the absolute URL (resolves relative URLs).
     *   2. Skip if null or blank.
     *   3. Validate the URL using UrlValidator (http/https scheme, valid host).
     *   4. Normalize the URL (strip fragment, trailing slash).
     *   5. If sameDomainOnly is true, check that the link is within the seed domain.
     *   6. Add to result list if not already present (dedupe).
     * 
     * @param doc The parsed HTML document
     * @param pageUrl The URL of the page being crawled (for absolute link resolution)
     * @param seedUrl The original seed URL (for same-domain checks)
     * @param sameDomainOnly If true, only keep links within the seed domain
     * @return List of valid, normalized, deduplicated discovered URLs
     */
    private List<String> extractLinks(Document doc, String pageUrl, String seedUrl, boolean sameDomainOnly) {
        List<String> result = new ArrayList<>();
        Elements anchors = doc.select("a[href]"); // Select all elements with an href attribute

        for (Element a : anchors) {
            // Get the absolute URL (resolves relative URLs against page URL)
            String absUrl = a.attr("abs:href");
            if (absUrl == null || absUrl.isBlank()) continue; // Skip null/empty hrefs

            // Validate the URL (scheme, host, disallowed extensions)
            if (!UrlValidator.isValid(absUrl)) continue;

            // Normalize the URL (strip fragment, trailing slash)
            String normalized = UrlValidator.normalize(absUrl);

            // If sameDomainOnly is true, check that the link is within the seed domain
            if (sameDomainOnly && !UrlValidator.isSameDomain(seedUrl, normalized)) continue;

            // Add to result list if not already present (dedupe by normalized URL)
            if (!result.contains(normalized)) {
                result.add(normalized);
            }
        }
        return result;
    }

    // ---- depth-encoding helpers: store "<depth>|<url>" as the queue entry ----

    /** Encode a URL and its depth into a single string for Redis queue storage. */
    private String encodeDepth(String url, int depth) {
        return depth + "|" + url;
    }

    /** Decode a queue entry string back into [depth, url] array. */
    private String[] decodeDepth(String entry) {
        int idx = entry.indexOf('|');
        if (idx < 0) return new String[]{"0", entry}; // Fallback if no pipe found
        return new String[]{entry.substring(0, idx), entry.substring(idx + 1)};
    }

    // ---- Public accessors for status checking ----

    /** Returns true if a crawl job is currently running. */
    public boolean isJobRunning() {
        return jobRunning.get();
    }

    /** Returns the seed URL of the current crawl job, or null if no job. */
    public String getCurrentSeedUrl() {
        return currentSeedUrl;
    }

    /** Returns when the current crawl job started, or null if no job. */
    public Instant getJobStartedAt() {
        return jobStartedAt;
    }

    /** Stop the current job and drain the queue. */
    public void stop() {
        jobRunning.set(false);
        store.clearQueue();
    }
}