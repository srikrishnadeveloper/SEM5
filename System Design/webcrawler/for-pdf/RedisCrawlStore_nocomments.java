package com.sscse.webcrawler.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sscse.webcrawler.model.PageCrawlResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;

@Component
public class RedisCrawlStore {

    private final ObjectMapper objectMapper;

    private final RedisTemplate<String, String> redisTemplate;

    @Value("${crawler.redis.queue-key}")
    private String queueKey;

    @Value("${crawler.redis.visited-key}")
    private String visitedKey;

    @Value("${crawler.redis.page-key-prefix}")
    private String pageKeyPrefix;

    @Value("${crawler.redis.stats-key}")
    private String statsKey;

    public RedisCrawlStore(RedisTemplate<String, String> redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
    }

    public void enqueue(String url) {
        redisTemplate.opsForList().leftPush(queueKey, url);
    }

    public String dequeue() {
        return redisTemplate.opsForList().rightPop(queueKey);
    }

    public long queueSize() {
        Long size = redisTemplate.opsForList().size(queueKey);
        return size == null ? 0 : size;
    }

    public List<String> peekQueue(long count) {
        return redisTemplate.opsForList().range(queueKey, 0, count - 1);
    }

    public void clearQueue() {
        redisTemplate.delete(queueKey);
    }

    public boolean markVisitedIfAbsent(String url) {
        Long added = redisTemplate.opsForSet().add(visitedKey, url);
        return added != null && added > 0;
    }

    public boolean isVisited(String url) {
        Boolean member = redisTemplate.opsForSet().isMember(visitedKey, url);
        return Boolean.TRUE.equals(member);
    }

    public Set<String> getVisited() {
        return redisTemplate.opsForSet().members(visitedKey);
    }

    public long visitedCount() {
        Long size = redisTemplate.opsForSet().size(visitedKey);
        return size == null ? 0 : size;
    }

    public void clearVisited() {
        redisTemplate.delete(visitedKey);
    }

    public void savePageResult(PageCrawlResult result) {
        try {
            String json = objectMapper.writeValueAsString(result);
            redisTemplate.opsForValue().set(pageKeyPrefix + result.getUrl(), json);
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize crawl result for " + result.getUrl(), e);
        }
    }

    public PageCrawlResult getPageResult(String url) {
        String json = redisTemplate.opsForValue().get(pageKeyPrefix + url);
        if (json == null) return null;
        try {
            return objectMapper.readValue(json, PageCrawlResult.class);
        } catch (Exception e) {
            return null;
        }
    }

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

    public long incrementCrawled() {
        Long v = redisTemplate.opsForValue().increment(statsKey + ":crawled");
        return v == null ? 0 : v;
    }

    public long incrementFailed() {
        Long v = redisTemplate.opsForValue().increment(statsKey + ":failed");
        return v == null ? 0 : v;
    }

    public long getCrawledCount() {
        String v = redisTemplate.opsForValue().get(statsKey + ":crawled");
        return v == null ? 0 : Long.parseLong(v);
    }

    public long getFailedCount() {
        String v = redisTemplate.opsForValue().get(statsKey + ":failed");
        return v == null ? 0 : Long.parseLong(v);
    }

    public void resetAll() {
        clearQueue();
        clearVisited();

        Set<String> pageKeys = redisTemplate.keys(pageKeyPrefix + "*");
        if (pageKeys != null && !pageKeys.isEmpty()) {
            redisTemplate.delete(pageKeys);
        }

        redisTemplate.delete(statsKey + ":crawled");
        redisTemplate.delete(statsKey + ":failed");
    }
}
