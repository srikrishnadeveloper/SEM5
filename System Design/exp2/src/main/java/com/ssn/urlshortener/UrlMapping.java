package com.ssn.urlshortener;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

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
