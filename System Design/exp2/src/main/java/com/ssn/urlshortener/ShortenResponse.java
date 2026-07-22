package com.ssn.urlshortener;

//send the response back to the client

public class ShortenResponse {

    private String shortUrl;

    //constuctor - create object and then immedialtey stores it
    public ShortenResponse(String shortUrl) {
        this.shortUrl = shortUrl;
    }

    public String getShortUrl() {
        return shortUrl;
    }


    //used to change the value stored in the db not 
    public void setShortUrl(String shortUrl) {
        this.shortUrl = shortUrl;
    }
}
