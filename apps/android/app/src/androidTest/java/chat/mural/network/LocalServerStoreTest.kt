// apps/android/app/src/androidTest/java/chat/mural/network/LocalServerStoreTest.kt
package chat.mural.network

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LocalServerStoreTest {
    @Test fun savesReadsAndClearsIndependentlyOfOtherPreferences() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = LocalServerStore(context, "mural_local_server_test")
        try {
            assertNull(store.read())
            store.save("192.168.1.20:8000")
            assertEquals("192.168.1.20:8000", store.read())
            store.save("  192.168.1.21:9000  ")
            assertEquals("192.168.1.21:9000", store.read())
            store.clear()
            assertNull(store.read())
        } finally { store.clear() }
    }
}
