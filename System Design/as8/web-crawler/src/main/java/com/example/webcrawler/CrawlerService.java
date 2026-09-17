package com.example.webcrawler;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.util.*;

@Service
public class CrawlerService {

    private final StringRedisTemplate redis;

    public CrawlerService(StringRedisTemplate redis) {
        this.redis = redis;
    }
 
    public Map<String, Object> crawl(String seedUrl, int maxPages) {

        validateUrl(seedUrl);

        String crawlId = UUID.randomUUID().toString();

        String queueKey = "crawler:" + crawlId + ":queue";
        String visitedKey = "crawler:" + crawlId + ":visited";
        String discoveredKey = "crawler:" + crawlId + ":discovered";
        String statusKey = "crawler:" + crawlId + ":status";

        // Add seed URL to discovered set
        redis.opsForSet().add(discoveredKey, seedUrl);

        // Add seed URL to queue
        redis.opsForList().rightPush(queueKey, seedUrl);

        redis.opsForHash().put(statusKey, "seedUrl", seedUrl);
        redis.opsForHash().put(statusKey, "status", "RUNNING");

        int processed = 0;
        int duplicateCount = 0;

        while (processed < maxPages) {

            String url = redis.opsForList().leftPop(queueKey);

            // Queue is empty
            if (url == null) {
                break;
            }

            // Double-check visited set
            if (Boolean.TRUE.equals(
                    redis.opsForSet().isMember(visitedKey, url))) {
                duplicateCount++;
                continue;
            }

            try {

                // Fetch web page
                Document document = Jsoup.connect(url)
                        .userAgent("Mozilla/5.0")
                        .timeout(5000)
                        .get();

                // Mark URL as visited
                redis.opsForSet().add(visitedKey, url);

                processed++;

                // Store basic crawling information
                redis.opsForHash().put(
                        statusKey,
                        "lastUrl",
                        url
                );

                // Extract links
                Elements links = document.select("a[href]");

                for (Element link : links) {

                    String discoveredUrl =
                            link.absUrl("href");

                    if (isAllowedUrl(discoveredUrl)) {
                        if (Boolean.TRUE.equals(
                                redis.opsForSet()
                                        .isMember(visitedKey, discoveredUrl))) {
                            duplicateCount++;
                            continue;
                        }

                        // SADD returns 1 only for a new URL
                        Long added =
                                redis.opsForSet()
                                        .add(discoveredKey, discoveredUrl);

                        if (added != null && added == 1) {

                            redis.opsForList()
                                    .rightPush(
                                            queueKey,
                                            discoveredUrl
                                    );
                        } else {
                            duplicateCount++;
                        }
                    }
                }

            } catch (Exception e) {

                redis.opsForHash().put(
                        statusKey,
                        "lastError",
                        e.getMessage()
                );
            }
        }

        redis.opsForHash().put(
                statusKey,
                "status",
                "COMPLETED"
        );

        Long visited =
                redis.opsForSet().size(visitedKey);

        Long discovered =
                redis.opsForSet().size(discoveredKey);

        Long queueSize =
                redis.opsForList().size(queueKey);

        Set<String> urlsCrawled = redis.opsForSet().members(visitedKey);
        if (urlsCrawled == null) {
            urlsCrawled = Collections.emptySet();
        }

        Map<String, Object> response = new LinkedHashMap<String, Object>();

        response.put("crawlId", crawlId);
        response.put("seedUrl", seedUrl);
        response.put("status", "COMPLETED");
        response.put("totalCrawlLimit", maxPages);
        response.put("numberOfCrawls", processed);
        response.put("numberOfWebsitesFound", discovered != null ? discovered : 0);
        response.put("duplicateWebsitesFound", duplicateCount);
        response.put("urlsCrawled", urlsCrawled);

        // Additional metrics
        response.put("processedPages", processed);
        response.put("discoveredUrls", discovered != null ? discovered : 0);
        response.put("visitedUrls", visited != null ? visited : 0);
        response.put("remainingQueue", queueSize != null ? queueSize : 0);

        return response;
    }

    private void validateUrl(String url) {

        try {

            URI uri = URI.create(url);

            String scheme = uri.getScheme();

            if (!"http".equalsIgnoreCase(scheme)
                    && !"https".equalsIgnoreCase(scheme)) {

                throw new IllegalArgumentException(
                        "Only HTTP and HTTPS URLs are allowed"
                );
            }

            if (uri.getHost() == null) {
                throw new IllegalArgumentException(
                        "Invalid URL"
                );
            }

        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {

            throw new IllegalArgumentException(
                    "Invalid URL: " + url
            );
        }
    }

    private boolean isAllowedUrl(String url) {

        try {

            URI uri = URI.create(url);

            String scheme = uri.getScheme();

            return ("http".equalsIgnoreCase(scheme)
                    || "https".equalsIgnoreCase(scheme))
                    && uri.getHost() != null;

        } catch (Exception e) {

            return false;
        }
    }
}