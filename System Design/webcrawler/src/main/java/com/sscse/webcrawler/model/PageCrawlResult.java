package com.sscse.webcrawler.model;

import java.time.Instant;
import java.util.List;

/**
 * Result/metadata captured for a single crawled page.
 * Persisted as a Redis hash for later retrieval via GET /api/crawler/results.
 * 
 * Fields:
 *   - url: The fully qualified URL that was crawled
 *   - status: CrawlStatus enum indicating what happened
 *   - title: HTML <title> tag content (if fetch succeeded)
 *   - depth: Crawling depth from seed (seed = 0, links from seed = 1, etc.)
 *   - linksFound: Number of valid discovered links on this page
 *   - discoveredLinks: List of valid, normalized URLs found on this page
 *   - errorMessage: If status is FAILED, the error message explaining why
 *   - crawledAt: Instant when this page was crawled
 */
public class PageCrawlResult {

    /** The fully qualified URL that was crawled. */
    private String url;

    /** CrawlStatus enum indicating what happened during crawling. */
    private CrawlStatus status;

    /** HTML <title> tag content (if fetch succeeded). May be null on failure. */
    private String title;

    /** Crawling depth from seed (seed = 0, direct links = 1, etc.). */
    private int depth;

    /** Number of valid discovered links on this page. */
    private int linksFound;

    /** List of valid, normalized URLs found on this page. */
    private List<String> discoveredLinks;

    /** If status is FAILED, the error message explaining why. Otherwise null. */
    private String errorMessage;

    /** Instant when this page was crawled. */
    private Instant crawledAt;

    /** Default constructor (required for JSON deserialization/Redis persistence). */
    public PageCrawlResult() {}

    /**
     * Full constructor for a successfully crawled page.
     * @param url The URL that was crawled
     * @param status CrawlStatus.CRAWLED
     * @param title HTML title tag content
     * @param depth Crawling depth from seed
     * @param linksFound Number of valid links discovered
     * @param discoveredLinks List of valid, normalized discovered URLs
     * @param crawledAt When the page was crawled
     */
    public PageCrawlResult(String url, CrawlStatus status, String title, int depth,
                           int linksFound, List<String> discoveredLinks, Instant crawledAt) {
        this.url = url;
        this.status = status;
        this.title = title;
        this.depth = depth;
        this.linksFound = linksFound;
        this.discoveredLinks = discoveredLinks;
        this.errorMessage = null;
        this.crawledAt = crawledAt;
    }

    /**
     * Full constructor for a failed crawl.
     * @param url The URL that failed
     * @param status CrawlStatus.FAILED
     * @param title null (no title on failure)
     * @param depth Crawling depth
     * @param linksFound 0 (no links extracted on failure)
     * @param discoveredLinks empty list
     * @param errorMessage Error message explaining the failure
     * @param crawledAt When the failure occurred
     */
    public PageCrawlResult(String url, CrawlStatus status, String title, int depth,
                           int linksFound, List<String> discoveredLinks,
                           String errorMessage, Instant crawledAt) {
        this.url = url;
        this.status = status;
        this.title = title;
        this.depth = depth;
        this.linksFound = linksFound;
        this.discoveredLinks = discoveredLinks;
        this.errorMessage = errorMessage;
        this.crawledAt = crawledAt;
    }

    /** Returns the URL that was crawled. */
    public String getUrl() { return url; }
    /** Sets the URL that was crawled. */
    public void setUrl(String url) { this.url = url; }

    /** Returns the crawl status. */
    public CrawlStatus getStatus() { return status; }
    /** Sets the crawl status. */
    public void setStatus(CrawlStatus status) { this.status = status; }

    /** Returns the HTML title tag content. */
    public String getTitle() { return title; }
    /** Sets the HTML title tag content. */
    public void setTitle(String title) { this.title = title; }

    /** Returns the crawling depth from seed. */
    public int getDepth() { return depth; }
    /** Sets the crawling depth from seed. */
    public void setDepth(int depth) { this.depth = depth; }

    /** Returns the number of valid discovered links. */
    public int getLinksFound() { return linksFound; }
    /** Sets the number of valid discovered links. */
    public void setLinksFound(int linksFound) { this.linksFound = linksFound; }

    /** Returns the list of valid, discovered URLs. */
    public List<String> getDiscoveredLinks() { return discoveredLinks; }
    /** Sets the list of valid, discovered URLs. */
    public void setDiscoveredLinks(List<String> discoveredLinks) { this.discoveredLinks = discoveredLinks; }

    /** Returns the error message if status is FAILED. */
    public String getErrorMessage() { return errorMessage; }
    /** Sets the error message if status is FAILED. */
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }

    /** Returns when the page was crawled. */
    public Instant getCrawledAt() { return crawledAt; }
    /** Sets when the page was crawled. */
    public void setCrawledAt(Instant crawledAt) { this.crawledAt = crawledAt; }
}