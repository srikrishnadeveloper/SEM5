package com.sscse.webcrawler.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Configuration
public class CrawlerExecutorConfig {

    @Value("${crawler.thread-pool-size:8}")
    private int poolSize;

    @Bean
    public ExecutorService crawlerExecutorService() {
        return Executors.newFixedThreadPool(poolSize);
    }
}
