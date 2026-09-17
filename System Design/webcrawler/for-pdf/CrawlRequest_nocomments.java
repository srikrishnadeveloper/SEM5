package com.sscse.webcrawler.model;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

public class CrawlRequest {

    @NotBlank(message = "seedUrl must not be blank")
    @Pattern(regexp = "^(https?)://.+", message = "seedUrl must start with http:// or https://")
    private String seedUrl;

    @Min(value = 0, message = "maxDepth must be >= 0")
    private int maxDepth = 2;

    @Min(value = 1, message = "maxPages must be >= 1")
    private int maxPages = 50;

    private boolean sameDomainOnly = true;

    public CrawlRequest() {}

    public CrawlRequest(String seedUrl, int maxDepth, int maxPages, boolean sameDomainOnly) {
        this.seedUrl = seedUrl;
        this.maxDepth = maxDepth;
        this.maxPages = maxPages;
        this.sameDomainOnly = sameDomainOnly;
    }

    public String getSeedUrl() { return seedUrl; }
    public void setSeedUrl(String seedUrl) { this.seedUrl = seedUrl; }

    public int getMaxDepth() { return maxDepth; }
    public void setMaxDepth(int maxDepth) { this.maxDepth = maxDepth; }

    public int getMaxPages() { return maxPages; }
    public void setMaxPages(int maxPages) { this.maxPages = maxPages; }

    public boolean isSameDomainOnly() { return sameDomainOnly; }
    public void setSameDomainOnly(boolean sameDomainOnly) { this.sameDomainOnly = sameDomainOnly; }
}
