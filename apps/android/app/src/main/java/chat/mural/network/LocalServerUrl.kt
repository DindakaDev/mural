package chat.mural.network

import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

/** Turns a user-entered "host:port" (or a full "http://host:port") into a
 * base URL APIClient can post against. Adds a trailing empty path segment
 * so baseUrl.newBuilder().addPathSegments(path) appends onto it instead
 * of replacing the last segment -- the same shape APIClient's own
 * API_BASE_URL is built with. */
fun parseLocalServerUrl(address: String): HttpUrl? {
    val trimmed = address.trim()
    if (trimmed.isEmpty()) return null
    val normalized = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) trimmed else "http://$trimmed"
    return normalized.toHttpUrlOrNull()?.newBuilder()?.addPathSegment("")?.build()
}
