package com.ssn.urlshortener;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Base64;
import java.util.Optional;

@RestController
public class UrlShortenerController {

    private final UrlMappingRepository repository;

    public UrlShortenerController(UrlMappingRepository repository) {
        this.repository = repository;
    }


    @PostMapping("/shorten")
    public ResponseEntity<ShortenResponse> shorten(@RequestBody ShortenRequest request) {

        String longUrl = request.getLongUrl();
        String shortCode = generateUniqueShortCode(longUrl);

        UrlMapping mapping = new UrlMapping(shortCode, longUrl);
        repository.save(mapping);

        String shortUrl = "http://localhost:8080/" + shortCode;
        ShortenResponse response = new ShortenResponse(shortUrl);

        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }


    @GetMapping("/{shortCode}")
    public ResponseEntity<Void> redirect(@PathVariable String shortCode) {

        Optional<UrlMapping> result = repository.findByShortCode(shortCode);

        if (result.isEmpty()) {
            return ResponseEntity.notFound().build();
        }

        String longUrl = result.get().getLongUrl();

        return ResponseEntity.status(HttpStatus.FOUND)
                .header(HttpHeaders.LOCATION, longUrl)
                .build();
    }


    private String generateUniqueShortCode(String longUrl) {

        String hash = sha256Base64(longUrl);
        int length = 6;
        String code = hash.substring(0, length);

        while (repository.findByShortCode(code).isPresent() && length < 8) {
            length++;
            code = hash.substring(0, length);
        }

        return code;
    }

    private String sha256Base64(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));

            return Base64.getUrlEncoder().withoutPadding().encodeToString(hashBytes);

        } catch (Exception e) {
            throw new RuntimeException("Error generating hash", e);
        }
    }
}
