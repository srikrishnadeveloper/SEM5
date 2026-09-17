package com.example.webcrawler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CrawlerController.class)
class CrawlerControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private CrawlerService crawlerService;

    @Test
    @DisplayName("Test REST Controller POST /api/crawl Response Telemetry")
    void testCrawlEndpointSuccess() throws Exception {
        List<String> mockCrawledList = List.of("https://example.com", "https://example.com/about");

        Map<String, Object> mockResponse = new LinkedHashMap<>();
        mockResponse.put("crawlId", "test-crawl-id-123");
        mockResponse.put("seedUrl", "https://example.com");
        mockResponse.put("status", "COMPLETED");
        mockResponse.put("totalCrawlLimit", 5);
        mockResponse.put("numberOfCrawls", 2);
        mockResponse.put("numberOfWebsitesFound", 5);
        mockResponse.put("duplicateWebsitesFound", 1);
        mockResponse.put("urlsCrawled", mockCrawledList);
        mockResponse.put("processedPages", 2);
        mockResponse.put("discoveredUrls", 5);
        mockResponse.put("visitedUrls", 2);
        mockResponse.put("remainingQueue", 3);

        when(crawlerService.crawl(anyString(), anyInt())).thenReturn(mockResponse);

        String jsonPayload = """
                {
                    "seedUrl": "https://example.com",
                    "maxPages": 5
                }
                """;

        System.out.println("\n========================================================");
        System.out.println("         CONTROLLER API TEST - REQUEST & RESPONSE       ");
        System.out.println("========================================================");
        System.out.println(" POST /api/crawl Payload:");
        System.out.println(jsonPayload.trim());
        System.out.println(" Response Telemetry:");
        System.out.println("  - Seed URL               : " + mockResponse.get("seedUrl"));
        System.out.println("  - Total Crawl Limit      : " + mockResponse.get("totalCrawlLimit"));
        System.out.println("  - Number of Crawls       : " + mockResponse.get("numberOfCrawls"));
        System.out.println("  - Number of Websites Found: " + mockResponse.get("numberOfWebsitesFound"));
        System.out.println("  - Duplicate Websites Found: " + mockResponse.get("duplicateWebsitesFound"));
        System.out.println("  - URLs Crawled List      : " + mockResponse.get("urlsCrawled"));
        System.out.println("========================================================\n");

        mockMvc.perform(post("/api/crawl")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(jsonPayload))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.crawlId").value("test-crawl-id-123"))
                .andExpect(jsonPath("$.seedUrl").value("https://example.com"))
                .andExpect(jsonPath("$.status").value("COMPLETED"))
                .andExpect(jsonPath("$.totalCrawlLimit").value(5))
                .andExpect(jsonPath("$.numberOfCrawls").value(2))
                .andExpect(jsonPath("$.numberOfWebsitesFound").value(5))
                .andExpect(jsonPath("$.duplicateWebsitesFound").value(1))
                .andExpect(jsonPath("$.urlsCrawled[0]").value("https://example.com"))
                .andExpect(jsonPath("$.processedPages").value(2))
                .andExpect(jsonPath("$.discoveredUrls").value(5))
                .andExpect(jsonPath("$.visitedUrls").value(2))
                .andExpect(jsonPath("$.remainingQueue").value(3));
    }
}
