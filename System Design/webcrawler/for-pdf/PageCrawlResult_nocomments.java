package com.sscse.webcrawler.model;

import java.time.Instant;
import java.util.List;

public class PageCrawlResult {

    private String url;

    private CrawlStatus status;

    private String title;

    private int depth;

    private int linksFound;

    private List<String> discoveredLinks;

    private String errorMessage;

    private Instant crawledAt;

    public PageCrawlResult() {}

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

    public String getUrl() { return url; }
    public void setUrl(String url) { this.url = url; }

    public CrawlStatus getStatus() { return status; }
    public void setStatus(CrawlStatus status) { this.status = status; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public int getDepth() { return depth; }
    public void setDepth(int depth) { this.depth = depth; }

    public int getLinksFound() { return linksFound; }
    public void setLinksFound(int linksFound) { this.linksFound = linksFound; }

    public List<String> getDiscoveredLinks() { return discoveredLinks; }
    public void setDiscoveredLinks(List<String> discoveredLinks) { this.discoveredLinks = discoveredLinks; }

    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }

    public Instant getCrawledAt() { return crawledAt; }
    public void setCrawledAt(Instant crawledAt) { this.crawledAt = crawledAt; }
}
