package com.sscse.webcrawler.model;

import java.time.Instant;

public class CrawlJobStatusResponse {

    private String jobId;

    private String seedUrl;

    private CrawlJobState state;

    private long queuedCount;

    private long visitedCount;

    private long crawledCount;

    private long failedCount;

    private Instant startedAt;

    private Instant updatedAt;

    public CrawlJobStatusResponse() {}

    public CrawlJobStatusResponse(String jobId, String seedUrl, CrawlJobState state,
                                  long queuedCount, long visitedCount, long crawledCount,
                                  long failedCount, Instant startedAt, Instant updatedAt) {
        this.jobId = jobId;
        this.seedUrl = seedUrl;
        this.state = state;
        this.queuedCount = queuedCount;
        this.visitedCount = visitedCount;
        this.crawledCount = crawledCount;
        this.failedCount = failedCount;
        this.startedAt = startedAt;
        this.updatedAt = updatedAt;
    }

    public String getJobId() { return jobId; }
    public void setJobId(String jobId) { this.jobId = jobId; }

    public String getSeedUrl() { return seedUrl; }
    public void setSeedUrl(String seedUrl) { this.seedUrl = seedUrl; }

    public CrawlJobState getState() { return state; }
    public void setState(CrawlJobState state) { this.state = state; }

    public long getQueuedCount() { return queuedCount; }
    public void setQueuedCount(long queuedCount) { this.queuedCount = queuedCount; }

    public long getVisitedCount() { return visitedCount; }
    public void setVisitedCount(long visitedCount) { this.visitedCount = visitedCount; }

    public long getCrawledCount() { return crawledCount; }
    public void setCrawledCount(long crawledCount) { this.crawledCount = crawledCount; }

    public long getFailedCount() { return failedCount; }
    public void setFailedCount(long failedCount) { this.failedCount = failedCount; }

    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }

    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
