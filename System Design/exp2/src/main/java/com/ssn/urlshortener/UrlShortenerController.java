
// heart of the project 
// contains two rest endpoint
// post/shorten - to generate short url
// get/{shortcode} -  controller search the db if found rediect 302 else 404 not found


// take google.com
// generate sha 256hash - why sha same input - same output,
// fast,diffuclt to rever,gnereate unique looking values


// covert into base 64 
// why base 64:
//     sha is hexadecimal ie 0-9 a-f
//     but base:
//     A-Z
//     a-z
//     0-9
//     so it create shorther readble ones
    
// take first 6 charcaters
// check db whether it exsit
// if exsit:
//     Ayirwc
// else:
//     AyirwcCx or AyirwcC


// databse:
//     table name = url_mapping
//     colums
//         id          =1
//         shortCode   =airwyc
//         longUrl     = https://google.com

// http status code
// post 201 created
// get 302 found
// 404 not found means not exist




package com.ssn.urlshortener;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

// import java.net.URI;
import java.nio.charset.StandardCharsets;
// Used while converting text into bytes before hashing.
import java.security.MessageDigest;
// used to genereate sha -256
import java.util.Base64;
// to covert the hash into base64
import java.util.Optional;
// may or may not find a url



@RestController //class handles http requests
public class UrlShortenerController { //controller

    private final UrlMappingRepository repository; //stores repositry olbject which is used to commiuncate with the db

    //construcor
    public UrlShortenerController(UrlMappingRepository repository) {
        this.repository = repository;
    }


    @PostMapping("/shorten") //first api
    public ResponseEntity<ShortenResponse> shorten(@RequestBody ShortenRequest request) {

        String longUrl = request.getLongUrl();
        String shortCode = generateUniqueShortCode(longUrl);
        //calls a function to generate a shortcode

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
//returns
//Status

// 201 Created

// Body

// {
// "shortUrl":
// "http://localhost:8080/Ayirwc"
// }                
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
            //creates sha-256 hashing object
            //sha256 works on byte not strings
            //utf-8bytes - > sha256-  hashbytes
            byte[] hashBytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));

            return Base64.getUrlEncoder().withoutPadding().encodeToString(hashBytes);

        } catch (Exception e) {
            throw new RuntimeException("Error generating hash", e);
        }
    }
}
