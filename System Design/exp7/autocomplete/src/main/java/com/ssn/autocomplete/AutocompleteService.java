package com.ssn.autocomplete;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Service
public class AutocompleteService {

    private final Trie trie = new Trie();
    private final StringRedisTemplate redis;

    public AutocompleteService(StringRedisTemplate redis) {
        this.redis = redis;
    }

    /** Load search terms from terms.csv at startup. */
    @PostConstruct
    public void init() throws Exception {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(getClass().getClassLoader().getResourceAsStream("terms.csv")))) {
            String line;
            br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length == 2) {
                    trie.insert(parts[0].trim(), Long.parseLong(parts[1].trim()));
                }
            }
        }
    }

    /** Return top-K suggestions for a prefix. Redis cache first, Trie on miss. */
    public List<String> search(String prefix, int k) {
        String key = "auto:" + prefix.toLowerCase() + ":" + k;

        // 1. Check Redis
        List<String> cached = redis.opsForList().range(key, 0, -1);
        if (cached != null && !cached.isEmpty()) {
            return cached;
        }

        // 2. Search Trie
        List<Suggestion> list = trie.findTopK(prefix, k);

        // 3. Build result
        List<String> result = new ArrayList<>();
        for (Suggestion s : list) {
            result.add(s.term + " (freq=" + s.freq + ")");
        }

        // 4. Cache with 5-minute TTL
        if (!result.isEmpty()) {
            redis.opsForList().rightPushAll(key, result);
            redis.expire(key, 300, TimeUnit.SECONDS);
        }

        return result;
    }

    /** Simple Trie with prefix search and top-K by frequency. */
    static class Trie {
        TrieNode root = new TrieNode();

        void insert(String word, long freq) {
            TrieNode node = root;
            for (char c : word.toLowerCase().toCharArray()) {
                node = node.child.computeIfAbsent(c, x -> new TrieNode()); //want to calirf even more
            }
            node.word = word;
            node.freq = freq;
        }

        List<Suggestion> findTopK(String prefix, int k) {
            TrieNode node = root;
            for (char c : prefix.toLowerCase().toCharArray()) {
                node = node.child.get(c);
                if (node == null) return new ArrayList<>();
            }

            List<Suggestion> list = new ArrayList<>();
            collect(node, list);

            list.sort((a, b) -> Long.compare(b.freq, a.freq));
            return list.size() > k ? list.subList(0, k) : list;
        }

        void collect(TrieNode node, List<Suggestion> list) {
            if (node == null) return;
            if (node.word != null) list.add(new Suggestion(node.word, node.freq));
            for (TrieNode child : node.child.values()) {
                collect(child, list);
            }
        }
    }

    static class TrieNode {
        Map<Character, TrieNode> child = new HashMap<>();
        String word;
        long freq;
    }

    static class Suggestion {
        String term;
        long freq;
        Suggestion(String term, long freq) {
            this.term = term;
            this.freq = freq;
        }
    }
}