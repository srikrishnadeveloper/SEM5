package com.sscse.webcrawler.util;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.util.Set;

public final class UrlValidator {

    private static final Set<String> DISALLOWED_EXTENSIONS = Set.of(
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".ico",
            ".pdf", ".zip", ".rar", ".mp4", ".mp3", ".woff", ".woff2", ".ttf"
    );

    private UrlValidator() {}

    public static boolean isValid(String url) {
        if (url == null || url.isBlank()) return false;

        try {
            URL parsed = URI.create(url).toURL();
            String protocol = parsed.getProtocol();

            if (!"http".equalsIgnoreCase(protocol) && !"https".equalsIgnoreCase(protocol)) {
                return false;
            }

            if (parsed.getHost() == null || parsed.getHost().isBlank()) {
                return false;
            }

            String path = parsed.getPath() == null ? "" : parsed.getPath().toLowerCase();

            for (String ext : DISALLOWED_EXTENSIONS) {
                if (path.endsWith(ext)) return false;
            }

            return true;

        } catch (MalformedURLException | IllegalArgumentException e) {
            return false;
        }
    }

    public static String normalize(String url) {
        try {
            URI uri = URI.create(url);

            URI noFragment = new URI(uri.getScheme(), uri.getAuthority(), uri.getPath(),
                    uri.getQuery(), null);

            String normalized = noFragment.toString();

            if (normalized.endsWith("/") && normalized.length() > 1) {
                normalized = normalized.substring(0, normalized.length() - 1);
            }

            return normalized;
        } catch (Exception e) {
            return url;
        }
    }

    public static boolean isSameDomain(String seedUrl, String candidateUrl) {
        try {
            String seedHost = URI.create(seedUrl).toURL().getHost();
            String candHost = URI.create(candidateUrl).toURL().getHost();

            return seedHost != null && seedHost.equalsIgnoreCase(candHost);
        } catch (Exception e) {
            return false;
        }
    }
}
