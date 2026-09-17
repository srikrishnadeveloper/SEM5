package com.sscse.webcrawler.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Thread pool used to run crawl jobs asynchronously so that the REST API
 * returns immediately (202 Accepted) while crawling continues in the background.
 * The pool size directly affects concurrency behaviour observed during JMeter load tests.
 */
@Configuration
public class CrawlerExecutorConfig {

    @Value("${crawler.thread-pool-size:8}")
    private int poolSize;

    @Bean
    public ExecutorService crawlerExecutorService() {
        return Executors.newFixedThreadPool(poolSize);
    }
}
