// 3. UrlMappingRepository.java

// This communicates with the database.

// Instead of writing SQL,

package com.ssn.urlshortener;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface UrlMappingRepository extends JpaRepository<UrlMapping, Long> { //JpaRepository<Entity, PrimaryKey>

    Optional<UrlMapping> findByShortCode(String shortCode);
    //return url mapping oject if found if not found nothing return optional insted of null this is by spring
    //optional is like a box that may or may not contain
    //find by -search the db
    //shortcode get matchs with shortcode present in entiry so spring autmoacilly search
    //(string shortcode) - is value to search inside the code
}
