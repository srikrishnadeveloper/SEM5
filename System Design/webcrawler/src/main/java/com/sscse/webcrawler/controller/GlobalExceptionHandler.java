package com.sscse.webcrawler.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Global exception handler for the webcrawler application.
 * Centralizes exception handling across all controllers.
 * 
 * Annotations:
 *   - @RestControllerAdvice: Applies the exception handling globally to all @RestController beans.
 *   - @ExceptionHandler: Maps specific exception types to handler methods.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * Handles MethodArgumentNotValidExceptions, which occur when @Validated bean validation fails.
     * For example, when POST /api/crawler/start receives a CrawlRequest with invalid fields.
     * 
     * Extracts field errors and returns them as a map of field name -> error message.
     * Returns HTTP 400 (Bad Request).
     * 
     * @param ex The MethodArgumentNotValidException containing validation error information
     * @return ResponseEntity with HTTP 400 and a map of field errors
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<?> handleValidation(MethodArgumentNotValidException ex) {
        // Build a LinkedHashMap to preserve the order of field errors
        Map<String, String> errors = new LinkedHashMap<>();

        // Iterate over all field errors from the validation result
        ex.getBindingResult().getFieldErrors().forEach(fe ->
                errors.put(fe.getField(), fe.getDefaultMessage()));

        // Return bad request response with the errors map
        return ResponseEntity.badRequest().body(Map.of("errors", errors));
    }

    /**
     * Handles IllegalArgumentExceptions, which are thrown explicitly by the application
     * (e.g., invalid seed URL in /api/crawler/start).
     * 
     * Returns the exception message in a simple map.
     * Returns HTTP 400 (Bad Request).
     * 
     * @param ex The IllegalArgumentException containing the error message
     * @return ResponseEntity with HTTP 400 and the exception message
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<?> handleIllegalArgument(IllegalArgumentException ex) {
        // Return bad request response with the exception message
        return ResponseEntity.badRequest().body(Map.of("message", ex.getMessage()));
    }
}