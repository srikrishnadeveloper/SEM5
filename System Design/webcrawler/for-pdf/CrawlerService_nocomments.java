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

@Service
public class CrawlerService {

    private static final Logger log = LoggerFactory.getLogger(CrawlerService.class);

    private final RedisCrawlStore store;

    private final ExecutorService executorService;

    @Value("${crawler.connect-timeout-ms:5000}")
    private int connectTimeoutMs;

    @Value("${crawler.user-agent:SSNCE-UCS3513-WebCrawler/1.0}")
    private String userAgent;

    private final AtomicBoolean jobRunning = new AtomicBoolean(false);

    private volatile String currentSeedUrl;

    private volatile Instant jobStartedAt;

    public CrawlerService(RedisCrawlStore store, ExecutorService executorService) {
        this.store = store;
        this.executorService = executorService;
    }

    public void startCrawl(CrawlRequest request) {
        if (!UrlValidator.isValid(request.getSeedUrl())) {
            throw new IllegalArgumentException("Invalid seed URL: " + request.getSeedUrl());
        }

        String normalizedSeed = UrlValidator.normalize(request.getSeedUrl());

        this.currentSeedUrl = normalizedSeed;
        this.jobStartedAt = Instant.now();

        this.jobRunning.set(true);

        store.enqueue(encodeDepth(normalizedSeed, 0));

        int maxDepth = request.getMaxDepth();
        int maxPages = request.getMaxPages();
        boolean sameDomainOnly = request.isSameDomainOnly();

        AtomicInteger pagesProcessed = new AtomicInteger(0);

        int workers = Math.min(4, Math.max(1, maxPages));
        for (int i = 0; i < workers; i++) {
            executorService.submit(() ->
                    crawlLoop(normalizedSeed, maxDepth, maxPages, sameDomainOnly, pagesProcessed));
        }
    }

    private void crawlLoop(String seedUrl, int maxDepth, int maxPages,
                           boolean sameDomainOnly, AtomicInteger pagesProcessed) {
        while (pagesProcessed.get() < maxPages) {
            String entry = store.dequeue();
            if (entry == null) {
                if (store.queueSize() == 0) break;
                continue;
            }

            String[] decoded = decodeDepth(entry);
            int depth = Integer.parseInt(decoded[0]);
            String url = decoded[1];

            if (depth > maxDepth) {
                log.debug("SKIP (depth {} exceeds budget {}): {}", depth, maxDepth, url);
                continue;
            }

            processUrl(url, depth, seedUrl, maxDepth, sameDomainOnly);

            pagesProcessed.incrementAndGet();
        }

        jobRunning.set(false);
    }

    private void processUrl(String url, int depth, String seedUrl, int maxDepth, boolean sameDomainOnly) {
        boolean isNew = store.markVisitedIfAbsent(url);
        if (!isNew) {
            log.debug("SKIP (already visited): {}", url);
            return;
        }

        try {
            Document doc = Jsoup.connect(url)
                    .userAgent(userAgent)
                    .timeout(connectTimeoutMs)
                    .get();

            List<String> discovered = extractLinks(doc, url, seedUrl, sameDomainOnly);

            PageCrawlResult result = new PageCrawlResult(
                    url, CrawlStatus.CRAWLED, doc.title(), depth,
                    discovered.size(), discovered, Instant.now());
            store.savePageResult(result);
            store.incrementCrawled();

            log.info("CRAWLED [{}] depth={} links={}", url, depth, discovered.size());

            if (depth < maxDepth) {
                for (String link : discovered) {
                    if (!store.isVisited(link)) {
                        store.enqueue(encodeDepth(link, depth + 1));
                    }
                }
            }

        } catch (Exception e) {
            PageCrawlResult failResult = new PageCrawlResult(
                    url, CrawlStatus.FAILED, null, depth, 0, List.of(),
                    e.getMessage(), Instant.now());
            store.savePageResult(failResult);
            store.incrementFailed();

            log.warn("FAILED [{}]: {}", url, e.getMessage());
        }
    }

    private List<String> extractLinks(Document doc, String pageUrl, String seedUrl, boolean sameDomainOnly) {
        List<String> result = new ArrayList<>();
        Elements anchors = doc.select("a[href]");

        for (Element a : anchors) {
            String absUrl = a.attr("abs:href");
            if (absUrl == null || absUrl.isBlank()) continue;

            if (!UrlValidator.isValid(absUrl)) continue;

            String normalized = UrlValidator.normalize(absUrl);

            if (sameDomainOnly && !UrlValidator.isSameDomain(seedUrl, normalized)) continue;

            if (!result.contains(normalized)) {
                result.add(normalized);
            }
        }
        return result;
    }

    private String encodeDepth(String url, int depth) {
        return depth + "|" + url;
    }

    private String[] decodeDepth(String entry) {
        int idx = entry.indexOf('|');
        if (idx < 0) return new String[]{"0", entry};
        return new String[]{entry.substring(0, idx), entry.substring(idx + 1)};
    }

    public boolean isJobRunning() {
        return jobRunning.get();
    }

    public String getCurrentSeedUrl() {
        return currentSeedUrl;
    }

    public Instant getJobStartedAt() {
        return jobStartedAt;
    }

    public void stop() {
        jobRunning.set(false);
        store.clearQueue();
    }
}
