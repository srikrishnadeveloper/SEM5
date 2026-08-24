
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

import org.springframework.data.redis.core.StringRedisTemplate;
// spring boot auto creates this bean once redis starter is on classpath, no config class needed
// StringRedisTemplate = "String" because our key(shortcode) and value(longurl) are both plain strings
// (full explanation of redis + this class is at the bottom of the file, didnt want to clutter imports)
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
    private final StringRedisTemplate redis; //talks to memurai/redis, key = shortcode value = longurl
    //this is basically our "cache" object, like a hashmap but it lives outside the app in memurai

    //construcor
    public UrlShortenerController(UrlMappingRepository repository, StringRedisTemplate redis) {
        this.repository = repository;
        this.redis = redis;
    }


    @PostMapping("/shorten") //first api
    // not touching redis anywhere in here on purpose, this is a write not a read
    // its already going to the db to save the mapping, nothing to "cache" yet since
    // nobody has looked it up so far - caching only kicks in on the first GET
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

        // cache-aside: check redis first before touching the db
        // opsForValue() = "operations for value" - redis can do lists/sets/hashes too
        // but we only need plain get/set so this is the simple string one
        String longUrl = redis.opsForValue().get(shortCode);
        // .get() just returns null if the key isnt there, it doesnt throw or crash

        if (longUrl != null) {
            // cache hit, straight from redis, no db call
            // this is the whole point of caching - skip the db completely when we can
            return ResponseEntity.status(HttpStatus.FOUND)
                    .header(HttpHeaders.LOCATION, longUrl)
                    .build();
        }

        // cache miss, fall back to db like before
        // (same jpa call exp2 always used, nothing changed here)
        Optional<UrlMapping> result = repository.findByShortCode(shortCode);

        if (result.isEmpty()) {
            // not in redis AND not in db = actually doesnt exist, real 404
            // note we never write anything to redis in this case, no point caching a 404
            return ResponseEntity.notFound().build();
        }

        longUrl = result.get().getLongUrl();

        // now store it in redis so next request for this shortcode is a hit
        // .set(key, value) - basically redis.put(shortCode, longUrl) if it was a hashmap
        // no expiry set here so it stays cached until memurai restarts
        redis.opsForValue().set(shortCode, longUrl);

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
        String code = hash.substring(0, length); // to get only first few charahcters

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


// ------------------------------------------------------------------------------
// what is redis (writing this down so i actually remember it later)
// ------------------------------------------------------------------------------

// redis is an in-memory key-value database
// key value just means two things stored together like a dictionary/hashmap
//     key   = shortcode      eg: a2KA_d
//     value = longurl        eg: https://google.com/search?q=redis+caching

// "in-memory" means everything lives in ram, not on disk like h2/mysql do
// thats WHY its so much faster - no disk read, no sql parsing, just a direct lookup
// downside: if redis restarts, everything cached is gone (we didnt set up persistence)

// memurai is literally just redis, but built to run on windows
// (actual redis doesnt officially support windows, memurai is the windows port of it)
// it runs as a background service on port 6379 by default, thats what
// spring.data.redis.host/port in application.properties points to

// StringRedisTemplate (the class we inject in the constructor)
//     spring's ready-made class for talking to redis, dont need to write raw redis commands
//     "String" because both our key and value are just plain strings, not objects/numbers
//     spring boot auto-creates this bean by itself once the redis starter dependency is
//     added to pom.xml, so we didnt need to write any extra RedisConfig.java class

// redis.opsForValue()
//     "operations for value" - just says im doing plain get/set stuff
//     redis also supports lists, sets, hashes etc but we dont need any of that here

// .get(shortCode)  -> returns the value if key exists, or null if it doesnt (cache miss)
// .set(shortCode, longUrl) -> saves key+value into redis so next .get() finds it (cache hit)

// cache-aside pattern:
//     1. check redis first for the shortcode
//     2. if found (HIT)  -> return it straight away, db is never touched
//     3. if not found (MISS) -> go to the db like exp2 always did,
//        then SAVE that result into redis before returning the response
//     4. next time the same shortcode is requested, its a HIT

// why this actually matters (numbers from my own testing):
//     db lookup (cache miss)  ~ 600-700 ms
//     redis lookup (cache hit) ~ 4-18 ms
//     so a cached request can be 30-100x+ faster than going to the db every time

// one thing to note: POST /shorten is NOT cached on purpose
// caching only makes sense for reads (GET), the POST already has to write to
// the db anyway to create the mapping in the first place, nothing to cache there

// ------------------------------------------------------------------------------
// storage limit + eviction policy (maxmemory / allkeys-lru)
// ------------------------------------------------------------------------------

// none of this lives in java code - its all set on the redis/memurai server itself,
// not the spring app, so nothing here needed to change for it

// maxmemory 20mb
//     caps how much ram redis is allowed to use for ALL cached keys combined
//     once it hits 20mb, redis has to make room before accepting new keys

// maxmemory-policy allkeys-lru
//     "allkeys" = policy applies to every key (not just ones with an expiry set)
//     "lru" = least recently used - when its full, redis kicks out whichever
//     key hasnt been read/written in the longest time, to make space for the new one
//     makes sense for us: old/rarely-visited short urls get evicted first,
//     popular ones (accessed often) stay cached longer

// set it like this (persists across memurai restarts):
//     memurai-cli CONFIG SET maxmemory 20mb
//     memurai-cli CONFIG SET maxmemory-policy allkeys-lru
//     memurai-cli CONFIG REWRITE      <- saves it into memurai.conf so it survives a restart

// tested this for real: wrote 300 keys x 100kb (~30mb total) into the cache,
// used_memory stayed capped at ~19.9mb the whole time and 114 of the 300 keys
// got evicted automatically - confirms the cap and lru eviction both actually work,
// not just that the CONFIG SET command returned OK
