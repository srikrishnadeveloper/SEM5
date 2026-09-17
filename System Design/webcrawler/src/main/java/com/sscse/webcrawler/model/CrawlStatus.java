package com.sscse.webcrawler.model;

/**
 * Lifecycle status of a single URL as it moves through the crawler.
 * Each crawled page gets one of these statuses.
 */
public enum CrawlStatus {
    /** URL is waiting in the queue to be crawled. */
    QUEUED,

    /** URL is currently being fetched/parsed. */
    CRAWLING,

    /** URL was successfully fetched and links were extracted. */
    CRAWLED,

    /** URL fetch failed (timeout, 404, connection error, etc.). */
    FAILED,

    /** URL was rejected by validation (invalid/disallowed) or already visited. */
    SKIPPED
}