package com.example.webcrawler;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class CrawlerController {

    private final CrawlerService crawlerService;

    public CrawlerController(CrawlerService crawlerService) {
        this.crawlerService = crawlerService;
    }
    

    @PostMapping("/crawl")
    public ResponseEntity<Map<String, Object>> crawl(
            @RequestBody CrawlRequest request) {

        Map<String, Object> result =
                crawlerService.crawl(
                        request.getSeedUrl(),
                        request.getMaxPages()
                );

        return ResponseEntity.ok(result);
    }
}