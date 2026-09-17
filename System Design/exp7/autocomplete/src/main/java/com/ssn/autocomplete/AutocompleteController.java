package com.ssn.autocomplete;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class AutocompleteController {

    private final AutocompleteService service;

    public AutocompleteController(AutocompleteService service) {
        this.service = service;
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Autocomplete service is up");
    }

    @GetMapping("/search")
    public ResponseEntity<List<String>> search(
            @RequestParam String prefix,
            @RequestParam(value = "k", defaultValue = "5") int k) {
        return ResponseEntity.ok(service.search(prefix, k));
    }
}