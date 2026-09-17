package com.sscse.webcrawler.util;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.util.Set;

/**
 * Validates and normalizes URLs before they are allowed into the crawl queue.
 * 
 * Only well-formed http/https URLs are accepted, and a small set of
 * non-content file extensions is rejected to avoid wasting crawl budget
 * on binaries (images, archives, stylesheets, etc.).
 * 
 * Design notes:
 * - DISALLOWED_EXTENSIONS covers common non-HTML content types.
 * - URL normalization (stripping fragments and trailing slashes) ensures
 *   that equivalent URLs dedupe correctly in the visited set.
 * - Same-domain checks use the hostname comparison, ignoring path differences.
 */
public final class UrlValidator {

    /** Set of file extensions that are disallowed from crawling.
     * These are typically non-content binaries that would waste crawl budget. */
    private static final Set<String> DISALLOWED_EXTENSIONS = Set.of(
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".ico",
            ".pdf", ".zip", ".rar", ".mp4", ".mp3", ".woff", ".woff2", ".ttf"
    );

    /** Private constructor - this is a utility class with only static methods. */
    private UrlValidator() {}

    /**
     * Returns true if the URL is syntactically valid, uses http/https protocol,
     * has a valid host, and is not a disallowed file type.
     * 
     * Steps:
     *   1. Check for null or blank input.
     *   2. Parse the URL and verify the protocol is http or https.
     *   3. Verify the host is present and not blank.
     *   4. Check the file extension against DISALLOWED_EXTENSIONS.
     *   5. Return true if all checks pass.
     * 
     * @param url The URL string to validate
     * @return true if the URL is valid and acceptable for crawling
     */
    public static boolean isValid(String url) {
        if (url == null || url.isBlank()) return false;

        try {
            // Parse the URL using URI.create().toURL() for validation
            URL parsed = URI.create(url).toURL();
            String protocol = parsed.getProtocol();

            // Check that the protocol is http or https
            if (!"http".equalsIgnoreCase(protocol) && !"https".equalsIgnoreCase(protocol)) {
                return false;
            }

            // Verify the host is present and not blank
            if (parsed.getHost() == null || parsed.getHost().isBlank()) {
                return false;
            }

            // Get the path (lowercase for case-insensitive extension comparison)
            String path = parsed.getPath() == null ? "" : parsed.getPath().toLowerCase();

            // Check against disallowed file extensions
            for (String ext : DISALLOWED_EXTENSIONS) {
                if (path.endsWith(ext)) return false;
            }

            // All checks passed
            return true;

        } catch (MalformedURLException | IllegalArgumentException e) {
            // Malformed URL or invalid argument - not acceptable for crawling
            return false;
        }
    }

    /**
     * Strips fragment (#...) identifiers and trailing slashes so equivalent URLs
     * dedupe correctly in the visited set.
     * 
     * Steps:
     *   1. Parse the URL into a URI.
     *   2. Create a new URI without the fragment (null 5th argument).
     *   3. Convert back to string.
     *   4. If the result ends with "/" and is longer than 1 character,
      *      remove the trailing slash.
     * 
     * @param url The URL string to normalize
     * @return A normalized version of the URL without fragments/trailing slashes,
     *         or the original URL if normalization fails
     */
    public static String normalize(String url) {
        try {
            // Parse the URL into its components
            URI uri = URI.create(url);

            // Create a new URI without the fragment identifier.
            // The null 5th argument means no fragment.
            URI noFragment = new URI(uri.getScheme(), uri.getAuthority(), uri.getPath(),
                    uri.getQuery(), null);

            // Convert back to string
            String normalized = noFragment.toString();

            // If the normalized URL ends with "/" and is longer than 1 character,
            // remove the trailing slash to ensure consistent formatting.
            if (normalized.endsWith("/") && normalized.length() > 1) {
                normalized = normalized.substring(0, normalized.length() - 1);
            }

            return normalized;
        } catch (Exception e) {
            // If normalization fails, return the original URL
            return url;
        }
    }

    /**
     * Check if two URLs are within the same domain.
     * 
     * Steps:
     *   1. Parse both URLs and extract their hosts.
     *   2. Compare the hosts case-insensitively.
     *   2. Return true if both hosts exist and match.
     * 
     * @param seedUrl The original seed URL (for reference/context)
     * @param candidateUrl The URL to check against the seed domain
     * @return true if both URLs share the same domain/host
     */
    public static boolean isSameDomain(String seedUrl, String candidateUrl) {
        try {
            // Extract the host from the seed URL
            String seedHost = URI.create(seedUrl).toURL().getHost();
            // Extract the host from the candidate URL
            String candHost = URI.create(candidateUrl).toURL().getHost();

            // Return true if both hosts are non-null and match case-insensitively
            return seedHost != null && seedHost.equalsIgnoreCase(candHost);
        } catch (Exception e) {
            // If anything goes wrong during parsing, return false
            return false;
        }
    }
}