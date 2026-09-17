package com.sscse.webcrawler.model;

import java.time.Instant;

/**
 * Response returned by GET /api/crawler/status.
 * Contains the current state of a crawl job plus statistics.
 */
public class CrawlJobStatusResponse {

    /** Unique identifier for the crawl job. */
    private String jobId;

    /** The seed URL that was used to start the crawl. */
    private String seedUrl;

    /** Current state of the crawl job. */
    private CrawlJobState state;

    /** Number of URLs currently waiting in the queue. */
    private long queuedCount;

    /** Number of unique URLs that have been visited. */
    private long visitedCount;

    /** Total number of pages successfully crawled. */
    private long crawledCount;

    /** Total number of pages that failed to crawl. */
    private long failedCount;

    /** Timestamp when the crawl job started. */
    private Instant startedAt;

    /** Timestamp when the crawl job was last updated. */
    private Instant updatedAt;

    /** Default constructor (required for JSON deserialization). */
    public CrawlJobStatusResponse() {}

    /**
     * Full constructor.
     * @param jobId Unique job identifier
     * @param seedUrl The seed URL used to start the crawl
     * @param state Current crawl state (PENDING, RUNNING, COMPLETED, STOPPED)
     * @param queuedCount Number of URLs waiting in the queue
     * @param visitedCount Number of unique URLs visited
     * @param crawledCount Number of successfully crawled pages
     * @param failedCount Number of failed crawl attempts
     * @param startedAt When the crawl job started
     * @param updatedAt Last time the job state was updated
     */
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

    /** Returns the unique job identifier. */
    public String getJobId() { return jobId; }
    /** Sets the unique job identifier. */
    public void setJobId(String jobId) { this.jobId = jobId; }

    /** Returns the seed URL. */
    public String getSeedUrl() { return seedUrl; }
    /** Sets the seed URL. */
    public void setSeedUrl(String seedUrl) { this.seedUrl = seedUrl; }

    /** Returns the current crawl state. */
    public CrawlJobState getState() { return state; }
    /** Sets the current crawl state. */
    public void setState(CrawlJobState state) { this.state = state; }

    /** Returns the number of URLs waiting in the queue. */
    public long getQueuedCount() { return queuedCount; }
    /** Sets the number of URLs waiting in the queue. */
    public void setQueuedCount(long queuedCount) { this.queuedCount = queuedCount; }

    /** Returns the number of unique URLs visited. */
    public long getVisitedCount() { return visitedCount; }
    /** Sets the number of unique URLs visited. */
    public void setVisitedCount(long visitedCount) { this.visitedCount = visitedCount; }

    /** Returns the total number of successfully crawled pages. */
    public long getCrawledCount() { return crawledCount; }
    /** Sets the total number of successfully crawled pages. */
    public void setCrawledCount(long crawledCount) { this.crawledCount = crawledCount; }

    /** Returns the total number of failed crawl attempts. */
    public long getFailedCount() { return failedCount; }
    /** Sets the total number of failed crawl attempts. */
    public void setFailedCount(long failedCount) { this.failedCount = failedCount; }

    /** Returns when the crawl job started. */
    public Instant getStartedAt() { return startedAt; }
    /** Sets when the crawl job started. */
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }

    /** Returns when the crawl job was last updated. */
    public Instant getUpdatedAt() { return updatedAt; }
    /** Sets when the crawl job was last updated. */
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}