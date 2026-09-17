package com.sscse.webcrawler.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sscse.webcrawler.model.PageCrawlResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;

/**
 * Thin wrapper around RedisTemplate that centralizes every Redis operation
 * used by the crawler.
 * 
 * Redis Key Conventions:
 *   Queue     -> Redis List   (crawler:queue)      LPUSH (enqueue) / RPOP (dequeue) => FIFO
 *   Visited   -> Redis Set    (crawler:visited)    SADD / SISMEMBER  => O(1) duplicate check
 *   Page data -> Redis String (crawler:page:{url}) SET/GET (JSON)   => crawl metadata cache
 *   Stats     -> Redis String (crawler:stats:*)    INCR              => atomic counters
 * 
 * Using Redis (instead of a plain in-memory HashSet/Queue) means the crawl
 * state survives an application restart and could be shared across multiple
 * crawler instances horizontally scaled behind a load balancer.
 */
@Component
public class RedisCrawlStore {

    /** Jackson ObjectMapper for JSON serialization/deserialization of PageCrawlResult. */
    private final ObjectMapper objectMapper;

    /** RedisTemplate for all Redis operations. */
    private final RedisTemplate<String, String> redisTemplate;

    /** Name of the Redis List used as the crawl URL queue. */
    @Value("${crawler.redis.queue-key}")
    private String queueKey;

    /** Name of the Redis Set used as the visited-URL set. */
    @Value("${crawler.redis.visited-key}")
    private String visitedKey;

    /** Prefix for Redis Keys storing individual page metadata. */
    @Value("${crawler.redis.page-key-prefix}")
    private String pageKeyPrefix;

    /** Name of the Redis String used as stats counters container. */
    @Value("${crawler.redis.stats-key}")
    private String statsKey;

    /**
     * Constructor-based dependency injection.
     * Sets up the ObjectMapper with JavaTimeModule for proper Instant serialization.
     * @param redisTemplate Spring Data Redis Template
     */
    public RedisCrawlStore(RedisTemplate<String, String> redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
    }

    // ---------- QUEUE (Redis List) ----------

    /**
     * Add a URL to the end of the crawl queue (FIFO).
     * Uses LPUSH which actually pushes to the head, but we treat it as FIFO
     * by using RPOP for dequeue. This is a common Redis queue pattern.
     * @param url The "depth|url" entry to enqueue
     */
    public void enqueue(String url) {
        redisTemplate.opsForList().leftPush(queueKey, url);
    }

    /**
     * Removes and returns the next URL to crawl (FIFO).
     * Uses RPOP (right pop) which returns the last-inserted element,
     * paired with leftPush creates a FIFO queue.
     * Returns null if the queue is empty.
     * @return The "depth|url" entry, or null if queue is empty
     */
    public String dequeue() {
        return redisTemplate.opsForList().rightPop(queueKey);
    }

    /**
     * Returns the current number of URLs waiting in the queue.
     * @return Queue size (0 if empty)
     */
    public long queueSize() {
        Long size = redisTemplate.opsForList().size(queueKey);
        return size == null ? 0 : size;
    }

    /**
     * Returns a list of URLs currently in the queue (for debugging/peeking).
     * @param count Maximum number of entries to return
     * @return List of "depth|url" entries
     */
    public List<String> peekQueue(long count) {
        return redisTemplate.opsForList().range(queueKey, 0, count - 1);
    }

    /**
     * Clear the entire crawl queue by deleting the Redis key.
     */
    public void clearQueue() {
        redisTemplate.delete(queueKey);
    }

    // ---------- VISITED SET (Redis Set) ----------

    /**
     * Atomically mark a URL as visited.
     * Returns true if the URL was newly added (i.e. not visited before).
     * Uses SADD which returns the number of new elements added.
     * @param url The URL to mark as visited
     * @return true if this was a new URL (not previously visited)
     */
    public boolean markVisitedIfAbsent(String url) {
        Long added = redisTemplate.opsForSet().add(visitedKey, url);
        return added != null && added > 0;
    }

    /**
     * Check if a URL has already been visited.
     * @param url The URL to check
     * @return true if the URL is in the visited set
     */
    public boolean isVisited(String url) {
        Boolean member = redisTemplate.opsForSet().isMember(visitedKey, url);
        return Boolean.TRUE.equals(member);
    }

    /**
     * Return all visited URLs as a Set.
     * @return Set of all visited URLs
     */
    public Set<String> getVisited() {
        return redisTemplate.opsForSet().members(visitedKey);
    }

    /**
     * Return the number of unique URLs that have been visited.
     * @return Visited count
     */
    public long visitedCount() {
        Long size = redisTemplate.opsForSet().size(visitedKey);
        return size == null ? 0 : size;
    }

    /**
     * Clear the entire visited set by deleting the Redis key.
     */
    public void clearVisited() {
        redisTemplate.delete(visitedKey);
    }

    // ---------- PAGE METADATA (Redis String, JSON-serialized) ----------

    /**
     * Save a PageCrawlResult to Redis as a JSON-serialized string.
     * The key is constructed as: {pageKeyPrefix}{url}
     * 
     * @param result The PageCrawlResult to persist
     * @throws RuntimeException if JSON serialization fails
     */
    public void savePageResult(PageCrawlResult result) {
        try {
            String json = objectMapper.writeValueAsString(result);
            redisTemplate.opsForValue().set(pageKeyPrefix + result.getUrl(), json);
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize crawl result for " + result.getUrl(), e);
        }
    }

    /**
     * Retrieve a PageCrawlResult from Redis by URL.
     * @param url The URL that was crawled
     * @return The PageCrawlResult, or null if not found
     */
    public PageCrawlResult getPageResult(String url) {
        String json = redisTemplate.opsForValue().get(pageKeyPrefix + url);
        if (json == null) return null;
        try {
            return objectMapper.readValue(json, PageCrawlResult.class);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Retrieve all PageCrawlResult entries from Redis.
     * Uses KEYS command to find all keys matching the page prefix pattern.
     * Note: In production, KEYS can be slow on large datasets; consider SCAN instead.
     * @return List of all PageCrawlResult entries
     */
    public List<PageCrawlResult> getAllPageResults() {
        Set<String> keys = redisTemplate.keys(pageKeyPrefix + "*");
        return keys == null ? List.of() : keys.stream()
                .map(k -> redisTemplate.opsForValue().get(k))
                .filter(java.util.Objects::nonNull)
                .map(json -> {
                    try {
                        return objectMapper.readValue(json, PageCrawlResult.class);
                    } catch (Exception e) {
                        return null;
                    }
                })
                .filter(java.util.Objects::nonNull)
                .toList();
    }

    // ---------- STATS (Redis counters) ----------

    /**
     * Increment the "crawled" counter and return the new value.
     * Uses Redis INCR for atomic increment.
     * @return The new crawled count value
     */
    public long incrementCrawled() {
        Long v = redisTemplate.opsForValue().increment(statsKey + ":crawled");
        return v == null ? 0 : v;
    }

    /**
     * Increment the "failed" counter and return the new value.
     * Uses Redis INCR for atomic increment.
     * @return The new failed count value
     */
    public long incrementFailed() {
        Long v = redisTemplate.opsForValue().increment(statsKey + ":failed");
        return v == null ? 0 : v;
    }

    /**
     * Return the current "crawled" counter value.
     * @return Number of successfully crawled pages
     */
    public long getCrawledCount() {
        String v = redisTemplate.opsForValue().get(statsKey + ":crawled");
        return v == null ? 0 : Long.parseLong(v);
    }

    /**
     * Return the current "failed" counter value.
     * @return Number of failed crawl attempts
     */
    public long getFailedCount() {
        String v = redisTemplate.opsForValue().get(statsKey + ":failed");
        return v == null ? 0 : Long.parseLong(v);
    }

    // ---------- RESET EVERYTHING ----------

    /**
     * Clear all crawl state: queue, visited set, page data, and stats.
     * Used by the DELETE /api/crawler/reset endpoint and by CrawlerService.stop().
     */
    public void resetAll() {
        clearQueue();
        clearVisited();

        // Delete all page metadata keys matching the prefix
        Set<String> pageKeys = redisTemplate.keys(pageKeyPrefix + "*");
        if (pageKeys != null && !pageKeys.isEmpty()) {
            redisTemplate.delete(pageKeys);
        }

        // Delete stats counters
        redisTemplate.delete(statsKey + ":crawled");
        redisTemplate.delete(statsKey + ":failed");
    }
}