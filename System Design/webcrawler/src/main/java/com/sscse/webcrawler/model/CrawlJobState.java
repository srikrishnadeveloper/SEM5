package com.sscse.webcrawler.model;

/**
 * Overall state of a crawl job.
 * A "job" is one "start crawl" request with a seed URL.
 * 
 * Possible states:
 *   - PENDING: No crawl has been started yet
 *   - RUNNING: A crawl is currently in progress
 *   - COMPLETED: The crawl finished naturally (queue empty, budget exhausted)
 *   - STOPPED: The crawl was manually stopped via the /stop endpoint
 */
public enum CrawlJobState {
    PENDING,
    RUNNING,
    COMPLETED,
    STOPPED
}