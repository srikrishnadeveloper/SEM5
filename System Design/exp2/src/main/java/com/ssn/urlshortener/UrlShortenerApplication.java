//starting point

// tells Spring Boot

// start the application
// scan components
// create beans
// start Tomcat server

// Execution begins here.

package com.ssn.urlshortener;

import org.springframework.boot.SpringApplication; //important annoitation tell the spring 
//this is main appilcation start everything from here
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class UrlShortenerApplication {

    public static void main(String[] args) {
        SpringApplication.run(UrlShortenerApplication.class, args);
        //main  wherever we use run or mvn spring :run it executes this
        
    }
}
