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
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;


//like in the mongodb @document like that
//with this annotation the spring create a table insted of thinkint this is just class like other

@Entity
public class UrlMapping {

    @Id //this field is primary key
    //this tells the db autmoicallty generate id 
    //and auto increment  
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String shortCode;
    private String longUrl;

    public UrlMapping() {
    }

    public UrlMapping(String shortCode, String longUrl) {
        this.shortCode = shortCode;
        this.longUrl = longUrl;
    }

    public Long getId() {// why long is because the the id can be long
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
