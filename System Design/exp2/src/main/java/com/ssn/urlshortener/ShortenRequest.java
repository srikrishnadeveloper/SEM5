package com.ssn.urlshortener;

public class ShortenRequest {

    private String longUrl; //to store the url entered by the user

    public String getLongUrl() {
        return longUrl; //varaible store the url enterd by the user getter
    }

    public void setLongUrl(String longUrl) {
        this.longUrl = longUrl; //setter
    }
}
