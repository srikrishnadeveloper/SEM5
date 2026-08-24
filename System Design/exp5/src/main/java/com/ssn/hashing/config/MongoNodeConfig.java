package com.ssn.hashing.config;

import com.ssn.hashing.hashing.ConsistentHashRing; //Imports the custom class responsible for ring calculations and node lookups.  
import com.mongodb.client.MongoClients;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

import java.util.LinkedHashMap;
import java.util.Map;
import jakarta.annotation.PostConstruct;

/**
 * Reads the "storage.nodes" property (name:host:port, comma separated),
 * opens one MongoTemplate per node, and registers every node on the
 * consistent hash ring. This class is the "wiring" between our simple
 * consistent hashing logic and real MongoDB instances running in Docker.
 */
@Configuration
public class MongoNodeConfig {
    // Reads the "storage.nodes" property (name:host:port, comma separated)
    // what do you mean by comma seprated?In the context of the "storage.nodes" property, "comma separated" means that multiple values (e.g., node1:localhost:27017,node2:localhost:27018) are separated by commas.
    @Value("${storage.nodes}")
    private String nodesConfig;

    // Reads the "storage.database" property (the name of the database to use on each node)
    @Value("${storage.database}")
    private String databaseName;

    @Value("${storage.virtualNodesPerNode:3}") //defaulting to 3 virtual nodes per physical node if not specified in the properties file
    private int virtualNodesPerNode;

    //3. State Storage and Spring Beans

    // name -> MongoTemplate, e.g. "node1" -> template connected to localhost:27017
    private final Map<String, MongoTemplate> nodeTemplates = new LinkedHashMap<>();
    //a dict that stores each node with its corresponding mongo template, 

    //makes it global 
    @Bean
    public ConsistentHashRing consistentHashRing() {
        return new ConsistentHashRing(virtualNodesPerNode);
    }

    @Bean
    public Map<String, MongoTemplate> nodeTemplates() {
        return nodeTemplates;
    }

    /**
     * On startup, parse "node1:localhost:27017,node2:localhost:27018,..."
     * and build a MongoTemplate + ring entry for each one.
     */
    @PostConstruct
    public void init() {
        ConsistentHashRing ring = consistentHashRing();
        String[] nodes = nodesConfig.split(",");
        for (String node : nodes) {
            String[] parts = node.trim().split(":");
            String name = parts[0];
            String host = parts[1];
            int port = Integer.parseInt(parts[2]);

            MongoTemplate template = buildTemplate(host, port);
            nodeTemplates.put(name, template);
            ring.addNode(name);
        }
    }

    /** Helper used both at startup and when a new node is added at runtime. */
    public MongoTemplate buildTemplate(String host, int port) {
        String uri = "mongodb://" + host + ":" + port;
        return new MongoTemplate(MongoClients.create(uri), databaseName);
    }
}
