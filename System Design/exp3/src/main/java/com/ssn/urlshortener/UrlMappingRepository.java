// 3. UrlMappingRepository.java

// This communicates with the database.

// Instead of writing SQL,

package com.ssn.urlshortener;

// swapped JpaRepository for MongoRepository when we moved off h2 to mongodb
// PrimaryKey type also changed from Long to String since mongo ids are ObjectId strings
import org.springframework.data.mongodb.repository.MongoRepository;
import java.util.Optional;

public interface UrlMappingRepository extends MongoRepository<UrlMapping, String> { //MongoRepository<Document, PrimaryKey>

    Optional<UrlMapping> findByShortCode(String shortCode);
    //return url mapping oject if found if not found nothing return optional insted of null this is by spring
    //optional is like a box that may or may not contain
    //find by -search the db
    //shortcode get matchs with shortcode present in entiry so spring autmoacilly search
    //(string shortcode) - is value to search inside the code
}
