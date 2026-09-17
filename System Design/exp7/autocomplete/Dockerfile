FROM eclipse-temurin:17-jdk-alpine
WORKDIR /app
COPY target/autocomplete-1.0.0.jar app.jar
COPY src/main/resources/terms.csv /app/terms.csv
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]