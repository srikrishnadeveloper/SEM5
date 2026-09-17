package com.example.webcrawler;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CrawlRequestTest {

    @Test
    void testGettersAndSetters() {
        CrawlRequest request = new CrawlRequest();
        
        // Default maxPages
        assertEquals(10, request.getMaxPages());

        request.setSeedUrl("https://example.com");
        request.setMaxPages(5);

        assertEquals("https://example.com", request.getSeedUrl());
        assertEquals(5, request.getMaxPages());
    }
}
