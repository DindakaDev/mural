// apps/android/app/src/test/java/chat/mural/network/LocalServerUrlTest.kt
package chat.mural.network

import org.junit.Assert.*
import org.junit.Test

class LocalServerUrlTest {
    @Test fun addsHttpSchemeWhenMissing() {
        val url = parseLocalServerUrl("192.168.1.20:8000")
        assertNotNull(url)
        assertEquals("http", url!!.scheme)
        assertEquals("192.168.1.20", url.host)
        assertEquals(8000, url.port)
    }

    @Test fun keepsAnExplicitScheme() {
        val url = parseLocalServerUrl("https://voice.local:9443")
        assertNotNull(url)
        assertEquals("https", url!!.scheme)
    }

    @Test fun endsWithATrailingPathSegmentForAddPathSegments() {
        val url = parseLocalServerUrl("192.168.1.20:8000")!!
        // Matches the shape APIClient.API_BASE_URL uses so
        // baseUrl.newBuilder().addPathSegments(path) appends correctly
        // instead of replacing the last segment.
        assertEquals("/", url.encodedPath)
    }

    @Test fun rejectsBlankOrUnparseableInput() {
        assertNull(parseLocalServerUrl(""))
        assertNull(parseLocalServerUrl("   "))
        assertNull(parseLocalServerUrl("not a url at all :::"))
    }

    @Test fun trimsWhitespace() {
        val url = parseLocalServerUrl("  192.168.1.20:8000  ")
        assertNotNull(url)
        assertEquals("192.168.1.20", url!!.host)
    }
}
