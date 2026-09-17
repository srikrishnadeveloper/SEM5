package com.sscse.webcrawler.model;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

/**
 * Request payload to start a new crawl job.
 * Sent as JSON via POST /api/crawler/start.
 * 
 * Fields:
 *   - seedUrl: The starting URL for the crawl (must use http:// or https://)
 *   - maxDepth: Maximum crawling depth (0 = only seed page, default 2)
 *   - maxPages: Maximum number of pages to crawl (default 50)
 *   - sameDomainOnly: If true, only follow links within the same domain
 */
public class CrawlRequest {

    /** The starting URL for the crawl. Must be valid http:// or https:// URL. */
    @NotBlank(message = "seedUrl must not be blank")
    @Pattern(regexp = "^(https?)://.+", message = "seedUrl must start with http:// or https://")
    private String seedUrl;

    /** Maximum crawling depth. 0 means only the seed page is crawled. Default: 2. */
    @Min(value = 0, message = "maxDepth must be >= 0")
    private int maxDepth = 2;

    /** Maximum number of pages to crawl. Default: 50. */
    @Min(value = 1, message = "maxPages must be >= 1")
    private int maxPages = 50;

    /** If true, only follow links whose host matches the seed URL's host. Default: true. */
    private boolean sameDomainOnly = true;

    /** Default constructor. */
    public CrawlRequest() {}

    /** Constructor with all fields. */
    public CrawlRequest(String seedUrl, int maxDepth, int maxPages, boolean sameDomainOnly) {
        this.seedUrl = seedUrl;
        this.maxDepth = maxDepth;
        this.maxPages = maxPages;
        this.sameDomainOnly = sameDomainOnly;
    }

    /** Returns the seed URL. */
    public String getSeedUrl() { return seedUrl; }
    /** Sets the seed URL. */
    public void setSeedUrl(String seedUrl) { this.seedUrl = seedUrl; }

    /** Returns the maximum crawling depth. */
    public int getMaxDepth() { return maxDepth; }
    /** Sets the maximum crawling depth. */
    public void setMaxDepth(int maxDepth) { this.maxDepth = maxDepth; }

    /** Returns the maximum number of pages to crawl. */
    public int getMaxPages() { return maxPages; }
    /** Sets the maximum number of pages to crawl. */
    public void setMaxPages(int maxPages) { this.maxPages = maxPages; }

    /** Returns whether to restrict crawling to the same domain only. */
    public boolean isSameDomainOnly() { return sameDomainOnly; }
    /** Sets whether to restrict crawling to the same domain only. */
    public void setSameDomainOnly(boolean sameDomainOnly) { this.sameDomainOnly = sameDomainOnly; }
}