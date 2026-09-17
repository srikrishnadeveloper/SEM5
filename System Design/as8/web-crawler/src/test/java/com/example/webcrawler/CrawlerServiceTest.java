package com.example.webcrawler;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.data.redis.core.HashOperations;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.SetOperations;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CrawlerServiceTest {

    @Mock
    private StringRedisTemplate redisTemplate;

    @Mock
    private SetOperations<String, String> setOperations;

    @Mock
    private ListOperations<String, String> listOperations;

    @Mock
    private HashOperations<String, Object, Object> hashOperations;

    private CrawlerService crawlerService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        when(redisTemplate.opsForSet()).thenReturn(setOperations);
        when(redisTemplate.opsForList()).thenReturn(listOperations);
        when(redisTemplate.opsForHash()).thenReturn(hashOperations);

        crawlerService = new CrawlerService(redisTemplate);
    }

    @Test
    @DisplayName("Test Invalid URL Scheme (Non-HTTP/HTTPS)")
    void testInvalidUrlSchemeThrowsException() {
        System.out.println("\n[TEST EXECUTION] Testing Invalid URL Scheme Validation...");
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> crawlerService.crawl("ftp://invalid-url.com", 5)
        );
        assertTrue(exception.getMessage().contains("Only HTTP and HTTPS URLs are allowed"));
        System.out.println("  => Exception correctly caught: " + exception.getMessage());
    }

    @Test
    @DisplayName("Test Malformed URL String")
    void testMalformedUrlThrowsException() {
        System.out.println("\n[TEST EXECUTION] Testing Malformed URL Input...");
        assertThrows(
                IllegalArgumentException.class,
                () -> crawlerService.crawl("not-a-valid-url", 5)
        );
        System.out.println("  => Exception correctly thrown for malformed URL.");
    }

    @Test
    @DisplayName("Test Crawl Statistics Output & Map Generation")
    void testCrawlInitializationAndCompletionWhenQueueEmpty() {
        String seedUrl = "https://example.com";

        // Mock Redis responses
        when(listOperations.leftPop(anyString())).thenReturn(null); // Queue empty
        when(setOperations.size(anyString())).thenReturn(1L);
        when(listOperations.size(anyString())).thenReturn(0L);
        when(setOperations.members(anyString())).thenReturn(Set.of("https://example.com"));

        Map<String, Object> result = crawlerService.crawl(seedUrl, 5);

        System.out.println("\n========================================================");
        System.out.println("              CRAWLER TEST RESULT SUMMARY               ");
        System.out.println("========================================================");
        System.out.println(" Crawl ID               : " + result.get("crawlId"));
        System.out.println(" Seed URL               : " + result.get("seedUrl"));
        System.out.println(" Status                 : " + result.get("status"));
        System.out.println(" Total Crawl Limit      : " + result.get("totalCrawlLimit"));
        System.out.println(" Number of Crawls       : " + result.get("numberOfCrawls"));
        System.out.println(" Number of Websites Found: " + result.get("numberOfWebsitesFound"));
        System.out.println(" Duplicate Websites Found: " + result.get("duplicateWebsitesFound"));
        System.out.println(" URLs Crawled List      : " + result.get("urlsCrawled"));
        System.out.println("========================================================\n");

        assertNotNull(result);
        assertEquals("COMPLETED", result.get("status"));
        assertEquals("https://example.com", result.get("seedUrl"));
        assertEquals(5, result.get("totalCrawlLimit"));
        assertEquals(0, result.get("numberOfCrawls"));
        assertEquals(1L, result.get("numberOfWebsitesFound"));
        assertEquals(0, result.get("duplicateWebsitesFound"));
        assertNotNull(result.get("urlsCrawled"));
        assertEquals(0, result.get("processedPages"));
        assertNotNull(result.get("crawlId"));

        verify(setOperations).add(contains(":discovered"), eq(seedUrl));
        verify(listOperations).rightPush(contains(":queue"), eq(seedUrl));
        verify(hashOperations).put(contains(":status"), eq("status"), eq("RUNNING"));
        verify(hashOperations).put(contains(":status"), eq("status"), eq("COMPLETED"));
    }
}
