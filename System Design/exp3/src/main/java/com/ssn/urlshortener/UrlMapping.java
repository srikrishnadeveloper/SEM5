// 2. UrlMapping.java

// This is the Entity.

// It represents one row in the database.

// Database

// id
// shortCode
// longUrl
// The @Entity annotation tells JPA to create this table automatically.


package com.ssn.urlshortener;


//jpa annotations jpa stands for java persistence api 
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;


//like in the mongodb @document like that
//with this annotation the spring create a collection insted of thinking this is just class like other

@Document(collection = "url_mapping")
public class UrlMapping {

    @Id
    private String id;

    private String shortCode;
    private String longUrl;

    public UrlMapping() {
    }

    public UrlMapping(String shortCode, String longUrl) {
        this.shortCode = shortCode;
        this.longUrl = longUrl;
    }

    public String getId() {
        return id;
    }

    public String getShortCode() {
        return shortCode;
    }

    public void setShortCode(String shortCode) {
        this.shortCode = shortCode;
    }

    public String getLongUrl() {
        return longUrl;
    }

    public void setLongUrl(String longUrl) {
        this.longUrl = longUrl;
    }
}
