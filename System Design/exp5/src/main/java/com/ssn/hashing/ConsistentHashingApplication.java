package com.ssn.hashing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.mongo.MongoAutoConfiguration;
import org.springframework.boot.autoconfigure.data.mongo.MongoDataAutoConfiguration;

// We exclude the default Mongo auto-configuration because this project manages
// its OWN set of MongoTemplates (one per storage node) instead of a single
// application.properties spring.data.mongodb.uri connection.
@SpringBootApplication(exclude = {MongoAutoConfiguration.class, MongoDataAutoConfiguration.class})
public class ConsistentHashingApplication {
    public static void main(String[] args) {
        SpringApplication.run(ConsistentHashingApplication.class, args);
    }
}
