// 2. UrlMapping.java

// This is the Entity.

// It represents one row in the database.

// Database

// id
// shortCode
// longUrl
// The @Entity annotation tells JPA to create this table automatically.


package com.ssn.urlshortener;


// switched from jpa to spring data mongodb (exp3 update - moved off h2 to mongo)
// funny enough the old comment right below here literally called this out before it happened
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;


//like in the mongodb @document like that
//with this annotation the spring create a table insted of thinkint this is just class like other
//(this comment aged well, we actually are on mongo now so @Document is the real deal here)

@Document(collection = "url_mapping") //mongo calls a "table" a collection instead
public class UrlMapping {

    @Id //this field is primary key
    //no @GeneratedValue needed anymore - mongo auto generates this by itself (a string ObjectId, not an auto-increment number)
    //thats also why the type changed from Long to String below
    private String id;

    private String shortCode;
    private String longUrl;

    public UrlMapping() {
    }

    public UrlMapping(String shortCode, String longUrl) {
        this.shortCode = shortCode;
        this.longUrl = longUrl;
    }

    public String getId() {// mongo ids are strings (ObjectId hex), not numbers like h2 was
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
