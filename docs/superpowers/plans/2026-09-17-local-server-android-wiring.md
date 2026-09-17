# Local Server Android Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Android app point its live-voice conversation at a local
server (the `serverlocal/` WebRTC server built in the companion plan)
instead of OpenAI, via a new `LOCAL_SERVER` conversation provider — with
no changes to the wire protocol, since the local server already speaks
the exact event vocabulary `LiveTransport.kt`/`MuralViewModel.kt` expect.

**Architecture:** `APIClient` already takes its `baseUrl` and credential
source as constructor parameters; this plan adds a `requiresAuth` flag and
an `APIClient.local(baseUrl)` factory so a second instance can point at a
LAN address with no OpenAI key. A new `ConversationProvider.LOCAL_SERVER`
enum value and a plain (non-secret) `LocalServerStore` let the user pick
this mode and enter a host:port in Settings. `MuralViewModel` picks the
right `APIClient` per provider in the same place it already picks between
`api` and the hosted client.

**Tech Stack:** Kotlin, Jetpack Compose, OkHttp, existing JUnit unit tests
+ Android instrumented tests (this project's established split — plain
JVM unit tests for Context-free logic, `androidTest` for anything backed
by `SharedPreferences`/`Context`).

**Spec:** `docs/superpowers/specs/2026-09-15-local-voice-server-design.md`
(see "Android client changes" — this plan supersedes that section's file
list with what's actually in the codebase today; see corrections below)

## Corrections to the spec's "Android client changes" section

The spec named `SettingsScreen.kt` as the single place to add UI, and
implied `APIClient`'s `MissingKey`/header logic needed a `requiresAuth`
flag threaded through `post()` — both still true. Two things the spec got
wrong, found while reading the actual code for this plan:
- The provider picker (`PERSONAL_KEY` vs `HOSTED_MINUTES`) lives in
  `AccountSheet.kt`, not `SettingsScreen.kt`, and is gated behind
  `hostedAvailable` (a managed-accounts/billing feature flag). Mixing a
  dev-only "local server" option into that billing-focused picker would
  hide it behind an unrelated flag. This plan instead adds Local Server
  as its own settings row + dialog in `SettingsScreen.kt`'s existing
  pattern (mirroring the "Use your own API key" row), and calls
  `selectConversationProvider(LOCAL_SERVER)` directly when an address is
  saved — no `AccountSheet.kt` changes needed at all.
- The spec didn't mention `teaching()` (`MuralViewModel.kt:431-439`,
  which backs lesson/meaning/lookup generation via `api.respond(...)`).
  The local server built so far has no `/responses` endpoint (that's the
  companion "Plan B"), so this plan explicitly makes `teaching()` throw
  `HostedFailure.Unavailable` when `LOCAL_SERVER` is selected, the same
  way it already does for `HOSTED_MINUTES` outside a live session —
  otherwise selecting local mode would silently keep sending lesson/lookup
  requests to OpenAI while the person believes they're fully local.

## Global Constraints

- No changes to `LiveTransport.kt`, `LiveTransport`'s WebRTC/audio setup,
  or the event vocabulary — the local server already speaks it exactly.
- `LocalServerStore` stores a plain host:port string with NO encryption —
  it is not a secret, unlike `CredentialStore`'s OpenAI key. Do not reuse
  `CredentialStore` for it (its `save()` rejects anything not shaped like
  an OpenAI key).
- `APIClient`'s existing `internal constructor(key: String?, client:
  OkHttpClient, baseUrl: HttpUrl)` (used by `APIClientTest.kt`) must keep
  working unchanged — any new constructor parameter needs a default so
  existing call sites don't break.
- Context-backed stores (`LocalServerStore`) are tested via `androidTest`
  (instrumented, needs a device/emulator) per this project's existing
  split — `CredentialStore`/`ConversationProviderStore` have no JVM unit
  tests either, only `AccountSessionStore` has an `androidTest`. Pure
  logic (URL parsing) goes in a separate function with a plain JUnit test.
- `MuralViewModel.kt` itself has no direct unit tests in this project
  (only its top-level pure functions, `errorMessageRes`/
  `errorNeedsKeySetup`, are tested) — Task 4's verification is running the
  full existing suite plus a Gradle compile check, consistent with how
  the rest of the ViewModel is already (not directly) tested.

---

## File Structure

```
apps/android/app/src/main/java/chat/mural/
  core/ConversationProviders.kt          # modify: add LOCAL_SERVER, extend canStart()
  network/LocalServerUrl.kt              # new: pure parseLocalServerUrl(String) -> HttpUrl?
  network/LocalServerStore.kt            # new: plain SharedPreferences store (no encryption)
  network/APIClient.kt                   # modify: requiresAuth flag + APIClient.local() factory
  MuralViewModel.kt                       # modify: state, init load, provider switch, cloudReady, teaching() guard, save/remove
  ui/SettingsScreen.kt                    # modify: Local Server row + dialog

apps/android/app/src/main/res/values/strings.xml     # modify: new strings (English)
apps/android/app/src/main/res/values-es/strings.xml  # modify: new strings (Spanish)

apps/android/app/src/test/java/chat/mural/
  core/ConversationProvidersTest.kt      # modify: update canStart() call sites, add LOCAL_SERVER case
  network/LocalServerUrlTest.kt          # new
  network/APIClientTest.kt               # modify: add local-mode test

apps/android/app/src/androidTest/java/chat/mural/network/
  LocalServerStoreTest.kt                 # new
```

---

### Task 1: `ConversationProviders.kt` — add `LOCAL_SERVER`

**Files:**
- Modify: `apps/android/app/src/main/java/chat/mural/core/ConversationProviders.kt:8` (enum), `:23` (`canStart`)
- Modify: `apps/android/app/src/test/java/chat/mural/core/ConversationProvidersTest.kt:22-30`

**Interfaces:**
- Produces: `enum class ConversationProvider { PERSONAL_KEY, HOSTED_MINUTES, LOCAL_SERVER }`,
  `ConversationProviderPolicy.canStart(choice: ConversationProvider, hasKey: Boolean, hasLocalServer: Boolean, hosted: HostedReadiness): Boolean`
  (note: `hasLocalServer` is inserted as the 3rd parameter, before `hosted` —
  every existing call site needs a new argument, not just an appended one).

- [ ] **Step 1: Write the failing test**

Add to `ConversationProvidersTest.kt`, and update the existing test's 5
call sites to pass a `hasLocalServer` argument (use `false` for the
existing assertions — they're not about local-server behavior — and add
one new assertion pair for `LOCAL_SERVER` itself):

```kotlin
    @Test fun selectionNeverFallsBackBetweenPersonalKeyAndHosted() {
        val ready = HostedReadiness("owner", 1, true)
        assertTrue(ConversationProviderPolicy.canStart(ConversationProvider.PERSONAL_KEY, true, false, HostedReadiness()))
        assertFalse(ConversationProviderPolicy.canStart(ConversationProvider.PERSONAL_KEY, false, false, ready))
        assertFalse(ConversationProviderPolicy.canStart(ConversationProvider.HOSTED_MINUTES, true, false, HostedReadiness()))
        assertTrue(ConversationProviderPolicy.canStart(ConversationProvider.HOSTED_MINUTES, false, false, ready))
        for (invalid in listOf(ready.copy(accountID = null), ready.copy(availableMilliseconds = 0), ready.copy(enabled = false), ready.copy(checking = true)))
            assertFalse(ConversationProviderPolicy.canStart(ConversationProvider.HOSTED_MINUTES, true, false, invalid))
    }

    @Test fun localServerCanStartOnlyWhenAnAddressIsSaved() {
        assertTrue(ConversationProviderPolicy.canStart(ConversationProvider.LOCAL_SERVER, false, true, HostedReadiness()))
        assertFalse(ConversationProviderPolicy.canStart(ConversationProvider.LOCAL_SERVER, false, false, HostedReadiness()))
        // hasKey/hosted readiness are irrelevant to this provider's own gate.
        assertTrue(ConversationProviderPolicy.canStart(ConversationProvider.LOCAL_SERVER, true, true, HostedReadiness("owner", 0, false)))
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.core.ConversationProvidersTest"`
Expected: FAIL — compile error, `canStart` doesn't accept 4 arguments yet
and `ConversationProvider.LOCAL_SERVER` doesn't exist.

- [ ] **Step 3: Update `ConversationProviders.kt`**

```kotlin
/** Choice is explicit. An unavailable provider never authorizes use of the other one. */
enum class ConversationProvider { PERSONAL_KEY, HOSTED_MINUTES, LOCAL_SERVER }
```

```kotlin
    fun canStart(choice: ConversationProvider, hasKey: Boolean, hasLocalServer: Boolean, hosted: HostedReadiness): Boolean =
        when (choice) {
            ConversationProvider.PERSONAL_KEY -> hasKey
            ConversationProvider.LOCAL_SERVER -> hasLocalServer
            ConversationProvider.HOSTED_MINUTES -> hosted.ready
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.core.ConversationProvidersTest"`
Expected: PASS (all tests in the file, including the 2 above)

- [ ] **Step 5: Commit**

```bash
git add apps/android/app/src/main/java/chat/mural/core/ConversationProviders.kt apps/android/app/src/test/java/chat/mural/core/ConversationProvidersTest.kt
git commit -m "Add LOCAL_SERVER conversation provider"
```

---

### Task 2: Local server address — parsing + storage

**Files:**
- Create: `apps/android/app/src/main/java/chat/mural/network/LocalServerUrl.kt`
- Create: `apps/android/app/src/test/java/chat/mural/network/LocalServerUrlTest.kt`
- Create: `apps/android/app/src/main/java/chat/mural/network/LocalServerStore.kt`
- Create: `apps/android/app/src/androidTest/java/chat/mural/network/LocalServerStoreTest.kt`

**Interfaces:**
- Produces: `parseLocalServerUrl(address: String): HttpUrl?` (top-level,
  package `chat.mural.network`), `LocalServerStore(context: Context)`
  with `read(): String?`, `save(address: String)`, `clear()`.

- [ ] **Step 1: Write the failing test for URL parsing**

```kotlin
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.network.LocalServerUrlTest"`
Expected: FAIL with a compile error — `parseLocalServerUrl` doesn't exist.

- [ ] **Step 3: Write `LocalServerUrl.kt`**

```kotlin
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.network.LocalServerUrlTest"`
Expected: PASS (5/5)

- [ ] **Step 5: Write `LocalServerStore.kt`** (no failing test first — this
  is a thin `SharedPreferences` wrapper identical in shape to
  `CredentialStore`'s own preferences plumbing, and this project's
  convention is to verify these via `androidTest`, written next)

```kotlin
package chat.mural.network

import android.content.Context

/** Stores the local voice server's address (host:port) in plain
 * SharedPreferences. Not a secret -- no encryption, unlike
 * CredentialStore's OpenAI key. */
class LocalServerStore internal constructor(
    context: Context,
    preferencesName: String,
) {
    constructor(context: Context) : this(context, PREFERENCES)

    private val preferences = context.applicationContext.getSharedPreferences(preferencesName, Context.MODE_PRIVATE)

    fun read(): String? = preferences.getString(ADDRESS, null)?.takeIf { it.isNotBlank() }

    fun save(address: String) {
        check(preferences.edit().putString(ADDRESS, address.trim()).commit())
    }

    fun clear() {
        check(preferences.edit().remove(ADDRESS).commit())
    }

    companion object {
        private const val PREFERENCES = "mural_local_server"
        private const val ADDRESS = "address"
    }
}
```

- [ ] **Step 6: Write the androidTest for `LocalServerStore`**

```kotlin
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
```

- [ ] **Step 7: Run the androidTest (device/emulator required)**

Run: `adb devices` first to confirm a device/emulator is attached. If
none is available, note that in your report rather than skipping this
step silently — this project's other `androidTest` files (e.g.
`AccountSessionStoreTest.kt`) have the same requirement, it isn't new to
this task.

Run: `cd apps/android && ./gradlew connectedDebugAndroidTest --tests "chat.mural.network.LocalServerStoreTest"`
Expected: PASS, if a device/emulator was available.

- [ ] **Step 8: Commit**

```bash
git add apps/android/app/src/main/java/chat/mural/network/LocalServerUrl.kt apps/android/app/src/test/java/chat/mural/network/LocalServerUrlTest.kt apps/android/app/src/main/java/chat/mural/network/LocalServerStore.kt apps/android/app/src/androidTest/java/chat/mural/network/LocalServerStoreTest.kt
git commit -m "Add local server address parsing and plain-preferences storage"
```

---

### Task 3: `APIClient.kt` — auth-optional local mode

**Files:**
- Modify: `apps/android/app/src/main/java/chat/mural/network/APIClient.kt:31-39` (class header/constructors), `:58-68` (`post()`), `:168-192` (companion object)
- Modify: `apps/android/app/src/test/java/chat/mural/network/APIClientTest.kt`

**Interfaces:**
- Consumes: nothing new from Task 1/2 (this task is independent of them).
- Produces: `APIClient.local(baseUrl: HttpUrl): APIClient` (companion
  factory), and a 4-argument `internal constructor(key: String?, client:
  OkHttpClient, baseUrl: HttpUrl, requiresAuth: Boolean)` for tests.

- [ ] **Step 1: Write the failing test**

Add to `APIClientTest.kt` (after the existing `missingKeyAndUnsafePathsNeverSendRequests` test):

```kotlin
    @Test fun localModeSendsNoAuthorizationHeaderAndNeverThrowsMissingKey() = runBlocking {
        val local = APIClient(null, OkHttpClient(), server.url("/v1/"), requiresAuth = false)
        server.enqueue(MockResponse().setBody("""{"session":{"id":"local-1"},"transport":{"type":"webrtc","sdp":"v=0\\r\\n"}}"""))
        val provider: LiveSessionProvider = local
        provider.createLiveSession(LiveSessionRequest("v=0", "Teaching policy"))
        val request = server.takeRequest()
        assertNull(request.getHeader("Authorization"))
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.network.APIClientTest"`
Expected: FAIL with a compile error — no 4-argument constructor accepting
`requiresAuth` exists yet.

- [ ] **Step 3: Modify `APIClient.kt`**

Change the class header and constructors (replaces the existing lines):

```kotlin
class APIClient private constructor(
    private val readCredential: () -> String?,
    private val client: OkHttpClient = defaultClient(),
    private val baseUrl: HttpUrl = API_BASE_URL,
    private val requiresAuth: Boolean = true,
) : TeachingClient, LiveSessionProvider {
    constructor(credentials: CredentialStore) : this(credentials::read)

    internal constructor(key: String?, client: OkHttpClient, baseUrl: HttpUrl) :
        this({ key }, client, baseUrl)

    internal constructor(key: String?, client: OkHttpClient, baseUrl: HttpUrl, requiresAuth: Boolean) :
        this({ key }, client, baseUrl, requiresAuth)
```

Change `post()`'s auth handling (the `key`/`Authorization` lines only —
everything else in the function is unchanged):

```kotlin
    suspend fun post(path: String, body: JsonObject): JsonObject {
        if (!VALID_PATH.matches(path) || path.contains("..") || path.startsWith('/')) {
            throw APIException.InvalidResponse
        }
        val key = if (requiresAuth) readCredential() ?: throw APIException.MissingKey else null
        val requestBuilder = Request.Builder()
            .url(baseUrl.newBuilder().addPathSegments(path).build())
            .header("Content-Type", JSON_MEDIA_TYPE.toString())
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
        key?.let { requestBuilder.header("Authorization", "Bearer $it") }
        val request = requestBuilder.build()
```

(The rest of `post()` — the `suspendCancellableCoroutine` block — is
unchanged; `request` is still the value it builds against.)

Add the factory to the `companion object` (alongside `API_BASE_URL` etc):

```kotlin
        fun local(baseUrl: HttpUrl): APIClient = APIClient({ null }, defaultClient(), baseUrl, requiresAuth = false)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/android && ./gradlew testDebugUnitTest --tests "chat.mural.network.APIClientTest"`
Expected: PASS — all existing tests in the file plus the new one. In
particular confirm `missingKeyAndUnsafePathsNeverSendRequests` still
passes unchanged (it uses the 3-argument internal constructor, which
defaults `requiresAuth` to `true`).

- [ ] **Step 5: Commit**

```bash
git add apps/android/app/src/main/java/chat/mural/network/APIClient.kt apps/android/app/src/test/java/chat/mural/network/APIClientTest.kt
git commit -m "Add APIClient.local() factory for auth-optional local-server mode"
```

---

### Task 4: `MuralViewModel.kt` — wire the provider through

**Files:**
- Modify: `apps/android/app/src/main/java/chat/mural/MuralViewModel.kt` at:
  - `:74` (state properties, near `hasKey`)
  - `:82-83` (store/client construction)
  - `:169-181` (init block's history/key load)
  - `:305-315` (`cloudReady()`)
  - `:423-426` (near `hostedClient()` — add `localApiClient()` here)
  - `:431-439` (`teaching()`)
  - `:557-567` (near `saveKey`/`deleteKey` — add `saveLocalServerAddress`/`removeLocalServerAddress`)
  - `:653` (`start()`'s provider switch)

**Interfaces:**
- Consumes: `ConversationProvider.LOCAL_SERVER` (Task 1),
  `parseLocalServerUrl` + `LocalServerStore` (Task 2),
  `APIClient.local(baseUrl)` (Task 3).
- Produces: `vm.hasLocalServer: Boolean`, `vm.localServerAddress: String`,
  `vm.saveLocalServerAddress(address: String)`,
  `vm.removeLocalServerAddress()` — these are what Task 5's UI calls.

This task has no new automated test (see Global Constraints — this
project doesn't unit-test `MuralViewModel` directly, only its top-level
pure functions). Verification is: the full existing test suite still
passes, plus a compile check, plus a careful manual read-through of the
diff against the checklist in Step 5.

- [ ] **Step 1: Add state properties**

In `MuralViewModel.kt:74`, right after `var hasKey by mutableStateOf(false); private set`:

```kotlin
    var hasKey by mutableStateOf(false); private set
    var hasLocalServer by mutableStateOf(false); private set
    var localServerAddress by mutableStateOf(""); private set
```

- [ ] **Step 2: Construct the store**

In `MuralViewModel.kt:82-83`, right after `private val credentials = CredentialStore(application)`:

```kotlin
    private val credentials = CredentialStore(application)
    private val localServerStore = LocalServerStore(application)
    private val api = APIClient(credentials)
```

- [ ] **Step 3: Load the saved address at startup**

In `MuralViewModel.kt:169-181`, change the `Pair` to a `Triple` and read the third element:

```kotlin
                val loaded = withContext(Dispatchers.IO) { Triple(repository.load(), credentials.hasKey, localServerStore.read()) }
                archive = loaded.first.archive
                val providers = providerStore.read(if (loaded.second) ConversationProvider.PERSONAL_KEY else ConversationProvider.HOSTED_MINUTES)
                hostedSessionIDs = providers.hostedIDs
                pendingHostedOwnerID = providers.pendingOwnerID
                accountChangeBlocked = providers.pendingOwnerID != null
                guests?.recoverAcknowledgedOwnerAtStartup(pendingHostedOwnerID,
                    clear = { owner -> clearAcknowledgedGuestMarker(owner) },
                    onFailure = { presentError(getApplication<Application>().getString(R.string.error_guest_secure_storage_unavailable)) })
                conversationProvider = providers.selection
                finalAssessmentTickets = ConversationProviderPolicy.recoveryTickets(loaded.first.finalAssessments, hostedSessionIDs)
                hasKey = loaded.second
                localServerAddress = loaded.third.orEmpty()
                hasLocalServer = !loaded.third.isNullOrBlank()
                storageReady = true
```

(Every other line in this block — the ones not shown as changed above —
stays exactly as it is today; only the first line changes shape from
`Pair` to `Triple`, and two new assignments are added after `hasKey =
loaded.second`.)

- [ ] **Step 4: Gate `cloudReady()` on a saved address**

In `MuralViewModel.kt:305-315`, add a branch after the existing
`PERSONAL_KEY` check:

```kotlin
    private fun cloudReady(): Boolean {
        if (!storageReady) { presentError(getApplication<Application>().getString(R.string.error_resolve_local_history_first)); return false }
        if (archive.preferences.aiConsentVersion != 1) {
            presentError(getApplication<Application>().getString(R.string.error_accept_ai_consent)); return false
        }
        val currentHosted = session?.id in hostedSessionIDs
        if (!currentHosted && conversationProvider == ConversationProvider.PERSONAL_KEY && !hasKey) {
            presentError(getApplication<Application>().getString(R.string.error_missing_key), needsKeySetup = true); return false
        }
        if (!currentHosted && conversationProvider == ConversationProvider.LOCAL_SERVER && !hasLocalServer) {
            presentError(getApplication<Application>().getString(R.string.error_missing_local_server)); return false
        }
        return true
    }
```

- [ ] **Step 5: Add `localApiClient()`**

Near `hostedClient()` (`MuralViewModel.kt:423-426`), add:

```kotlin
    private fun localApiClient(): APIClient {
        val url = parseLocalServerUrl(localServerAddress) ?: throw APIClient.APIException.InvalidResponse
        return APIClient.local(url)
    }
```

(`parseLocalServerUrl` returning `null` here is a defensive fallback, not
an expected path — `saveLocalServerAddress` in Step 7 validates the
address before it's ever stored, so `localServerAddress` should always
parse by the time this runs.)

- [ ] **Step 6: Guard `teaching()` for local mode**

In `MuralViewModel.kt:431-439`, change the throw condition:

```kotlin
    private suspend fun teaching(localID: String?, purpose: HelperPurpose, logicalID: String,
        instructions: String, input: String, schema: JsonObject? = null, search: Boolean = false): APIResult {
        if (archive.preferences.aiConsentVersion != 1) throw HostedFailure.Unavailable
        if (localID != null && localID in hostedSessionIDs) {
            return hostedBindings.respond(localID, purpose, logicalID, instructions, input, schema, search)
        }
        if (localID == null && (conversationProvider == ConversationProvider.HOSTED_MINUTES || conversationProvider == ConversationProvider.LOCAL_SERVER)) throw HostedFailure.Unavailable
        return api.respond(instructions, input, schema, search, purpose)
    }
```

(This deliberately reuses `HostedFailure.Unavailable` rather than adding
a new exception type — the local server has no `/responses` endpoint yet,
so lesson/meaning/lookup requests report "unavailable" the same way they
already do for `HOSTED_MINUTES` outside a live session, instead of
silently falling through to OpenAI's real API while the person believes
they're in local mode.)

- [ ] **Step 7: Add save/remove functions**

Near `saveKey`/`deleteKey` (`MuralViewModel.kt:557-567`), add:

```kotlin
    fun saveLocalServerAddress(address: String) {
        if (isRunning) return
        val trimmed = address.trim()
        if (parseLocalServerUrl(trimmed) == null) {
            notice = getApplication<Application>().getString(R.string.error_invalid_local_server); return
        }
        localServerStore.save(trimmed)
        localServerAddress = trimmed
        hasLocalServer = true
        selectConversationProvider(ConversationProvider.LOCAL_SERVER)
        notice = getApplication<Application>().getString(R.string.notice_local_server_saved)
    }
    fun removeLocalServerAddress() {
        if (isRunning) return
        localServerStore.clear()
        hasLocalServer = false
        localServerAddress = ""
    }
```

- [ ] **Step 8: Route `start()`'s provider switch through `LOCAL_SERVER`**

In `MuralViewModel.kt:653`, change the `if/else` to a `when` — the
`HOSTED_MINUTES` branch's body is copied verbatim, unchanged:

```kotlin
                val provider: LiveSessionProvider = when (choice) {
                    ConversationProvider.PERSONAL_KEY -> api
                    ConversationProvider.LOCAL_SERVER -> localApiClient()
                    ConversationProvider.HOSTED_MINUTES -> {
                        val owner = requireHostedOwner()
                        if (selectedAccount.busy || owner.accountID != hostedReadiness.accountID) throw HostedFailure.SignInRequired
                        val hosted = hostedClient(owner.accountID)
                        val balance = hostedBalance(owner)
                        if (!balance.canStartConversation || !hosted.available()) throw HostedFailure.Unavailable
                        // Commit provider provenance and the unresolved-owner marker before making a paid create.
                        hostedSessionIDs = hostedSessionIDs + id
                        pendingHostedOwnerID = owner.accountID
                        withContext(NonCancellable) { providerStore.markHosted(id, owner.accountID) }
                        object : LiveSessionProvider {
                            override suspend fun createLiveSession(request: LiveSessionRequest): LiveSessionConnection {
                                val result = hosted.createLiveSession(request.copy(requestedMilliseconds = archive.preferences.sessionMinutes * 60_000L))
                                val lease = result.lease as? HostedAPIClient.HostedLease ?: throw HostedFailure.InvalidResponse
                                withContext(NonCancellable + Dispatchers.Main.immediate) {
                                    hostedBindings.bind(id, owner.accountID, binding(lease))
                                    if (session?.id != id || state != "connecting") {
                                        hostedBindings.ended(id); reconcileHostedSessions()
                                    }
                                }
                                return result
                            }
                        }
                    }
                }
```

Also update `start()`'s existing call to `ConversationProviderPolicy.canStart` (a few lines earlier in the same function) to pass `hasLocalServer`:

```kotlin
        if (!ConversationProviderPolicy.canStart(choice, hasKey, hasLocalServer, hostedReadiness)) {
```

- [ ] **Step 9: Run the full existing test suite**

Run: `cd apps/android && ./gradlew testDebugUnitTest`
Expected: PASS — every existing test, unchanged behavior for
`PERSONAL_KEY`/`HOSTED_MINUTES` paths. (`MuralViewModel` itself has no
direct unit test to run here, per this project's existing convention —
this run is confirming no regression in what IS tested:
`ConversationProvidersTest`, `APIClientTest`,
`errorMessageRes`/`errorNeedsKeySetup` in `MuralViewModelTest`, etc.)

- [ ] **Step 10: Compile check**

Run: `cd apps/android && ./gradlew compileDebugKotlin`
Expected: BUILD SUCCESSFUL — confirms the `when` in Step 8 is exhaustive
and every call site compiles.

- [ ] **Step 11: Self-review against this checklist**

Before committing, re-read the diff and confirm:
- `teaching()`'s new condition uses `||`, not `&&` (both providers should
  throw, not only both-at-once).
- The `when` in Step 8 has no `else` branch and still compiles — if it
  doesn't compile, `ConversationProvider` gained a case this task didn't
  handle; do not add a bare `else -> api` to paper over it, report BLOCKED
  instead.
- `localApiClient()` is called fresh inside `start()`'s `when`, not cached
  as a `val` at class level — this is deliberate: rebuilding it from the
  current `localServerAddress` each time means a changed-and-resaved
  address takes effect on the very next `start()` without needing to
  reconstruct anything else.

- [ ] **Step 12: Commit**

```bash
git add apps/android/app/src/main/java/chat/mural/MuralViewModel.kt
git commit -m "Wire LOCAL_SERVER provider through MuralViewModel"
```

---

### Task 5: `SettingsScreen.kt` — Local Server UI + strings

**Files:**
- Modify: `apps/android/app/src/main/java/chat/mural/ui/SettingsScreen.kt` at:
  - `:72-79` (dialog state vars, near `keyDialog`)
  - `:124-149` (insert a new `item { }` block right after the existing "Advanced" one)
  - `:216` (dialog invocation, near `if (keyDialog) KeyDialog(...)`)
  - `:233-235` (confirm-delete dialog, near the `deleteKey` one)
  - `:279` (new `LocalServerDialog` composable, right after `KeyDialog`)
- Modify: `apps/android/app/src/main/res/values/strings.xml` (English)
- Modify: `apps/android/app/src/main/res/values-es/strings.xml` (Spanish)

**Interfaces:**
- Consumes: `vm.hasLocalServer`, `vm.localServerAddress`,
  `vm.saveLocalServerAddress(address)`, `vm.removeLocalServerAddress()`
  (Task 4).

No new automated test — this project has no Compose UI tests for
`SettingsScreen` (confirmed: no `SettingsScreenTest` exists anywhere in
the repo). Verification is a Gradle compile check plus a manual run
(Step 5).

- [ ] **Step 1: Add dialog state**

In `SettingsScreen.kt:73-74`, right after `var keyDialog by rememberSaveable { mutableStateOf(false) }`:

```kotlin
    var keyDialog by rememberSaveable { mutableStateOf(false) }
    var localServerDialog by rememberSaveable { mutableStateOf(false) }
    var deleteKey by rememberSaveable { mutableStateOf(false) }
    var deleteLocalServer by rememberSaveable { mutableStateOf(false) }
```

- [ ] **Step 2: Add the settings row**

Insert a new `item { }` block in the `LazyColumn`, right after the
existing "Advanced" (BYOK) `item { }` block that ends at
`SettingsScreen.kt:149`:

```kotlin
            item {
                SettingsGroup(stringResource(R.string.settings_local_server_title),
                    stringResource(R.string.settings_local_server_footer)) {
                    SettingsRow(stringResource(if (vm.hasLocalServer) R.string.settings_local_server_change else R.string.settings_local_server_add),
                        enabled = !vm.isRunning, tint = MuralColors.Secondary, chevron = true,
                        modifier = Modifier.testTag("advanced-local-server"), onClick = { localServerDialog = true })
                    if (vm.hasLocalServer) {
                        SettingsDivider()
                        Text(vm.localServerAddress, style = MaterialTheme.typography.bodySmall,
                            color = MuralColors.Secondary, modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp))
                        SettingsDivider()
                        SettingsRow(stringResource(R.string.settings_local_server_remove), enabled = !vm.isRunning,
                            tint = MuralColors.Red, onClick = { deleteLocalServer = true })
                    }
                }
            }
```

- [ ] **Step 3: Wire the dialogs**

Right after `SettingsScreen.kt:216`'s `if (keyDialog) KeyDialog(vm, onDismiss = { keyDialog = false })`:

```kotlin
    if (localServerDialog) LocalServerDialog(vm, onDismiss = { localServerDialog = false })
```

Right after `SettingsScreen.kt:233-235`'s `deleteKey` `ConfirmDialog` call:

```kotlin
    if (deleteLocalServer) ConfirmDialog(stringResource(R.string.settings_delete_local_server_confirm_title), stringResource(R.string.settings_delete_local_server_confirm_message), stringResource(R.string.common_delete), {
        vm.removeLocalServerAddress(); deleteLocalServer = false
    }, { deleteLocalServer = false })
```

- [ ] **Step 4: Add the `LocalServerDialog` composable**

Right after the `KeyDialog` function ends (`SettingsScreen.kt`, the
function starting at line 279 — add this one immediately below its
closing brace):

```kotlin
@Composable
private fun LocalServerDialog(vm: MuralViewModel, onDismiss: () -> Unit) {
    var address by remember { mutableStateOf(vm.localServerAddress) }
    Dialog(onDismissRequest = onDismiss) {
        Surface(shape = RoundedCornerShape(28.dp), color = MuralColors.Surface) {
            Column(Modifier.padding(22.dp), verticalArrangement = Arrangement.spacedBy(15.dp)) {
                Text(stringResource(R.string.settings_local_server_dialog_title), style = MaterialTheme.typography.headlineMedium)
                Text(stringResource(R.string.settings_local_server_dialog_note), color = MuralColors.Secondary)
                MuralTextField(
                    address,
                    { address = it.take(200) },
                    Modifier.fillMaxWidth().testTag("local-server-input"),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, autoCorrectEnabled = false),
                    label = { Text(stringResource(R.string.settings_local_server_dialog_field_label)) },
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    MuralTextButton(onClick = onDismiss) { Text(stringResource(R.string.common_cancel)) }
                    Button(onClick = { vm.saveLocalServerAddress(address); onDismiss() }, enabled = address.isNotBlank()) { Text(stringResource(R.string.common_save)) }
                }
            }
        }
    }
}
```

- [ ] **Step 5: Add strings**

In `apps/android/app/src/main/res/values/strings.xml`, right after
`error_missing_key` (line 21):

```xml
    <string name="error_missing_key">Add your OpenAI key in Settings to begin.</string>
    <string name="error_missing_local_server">Add your local server address in Settings to begin.</string>
    <string name="error_invalid_local_server">Enter a valid address, like 192.168.1.20:8000.</string>
```

Right after `settings_backup_imported_toast` or anywhere in the settings
strings block — placed here next to the existing key-dialog strings
(right after `settings_key_dialog_field_label`, line 210):

```xml
    <string name="settings_key_dialog_field_label">New key</string>
    <string name="settings_local_server_title">Local server</string>
    <string name="settings_local_server_footer">Connect to a Whisper/Ollama/Kokoro server on your own network instead of OpenAI.</string>
    <string name="settings_local_server_add">Add server address</string>
    <string name="settings_local_server_change">Change server address</string>
    <string name="settings_local_server_remove">Remove server address</string>
    <string name="settings_delete_local_server_confirm_title">Remove the local server address?</string>
    <string name="settings_delete_local_server_confirm_message">You\'ll need to add it again to use local mode.</string>
    <string name="settings_local_server_dialog_title">Local server address</string>
    <string name="settings_local_server_dialog_note">Stored on this device only, not encrypted -- this isn\'t a secret, just an address on your network.</string>
    <string name="settings_local_server_dialog_field_label">Host:port</string>
```

And a notice string, near `notice_key_saved`:

```xml
    <string name="notice_local_server_saved">Local server address saved.</string>
```

In `apps/android/app/src/main/res/values-es/strings.xml`, the matching
Spanish strings in the same relative positions:

```xml
    <string name="error_missing_local_server">Añade la dirección de tu servidor local en Ajustes para empezar.</string>
    <string name="error_invalid_local_server">Ingresa una dirección válida, como 192.168.1.20:8000.</string>
```

```xml
    <string name="settings_local_server_title">Servidor local</string>
    <string name="settings_local_server_footer">Conecta a un servidor Whisper/Ollama/Kokoro en tu propia red en vez de OpenAI.</string>
    <string name="settings_local_server_add">Agregar dirección del servidor</string>
    <string name="settings_local_server_change">Cambiar dirección del servidor</string>
    <string name="settings_local_server_remove">Eliminar dirección del servidor</string>
    <string name="settings_delete_local_server_confirm_title">¿Eliminar la dirección del servidor local?</string>
    <string name="settings_delete_local_server_confirm_message">Tendrás que agregarla de nuevo para usar el modo local.</string>
    <string name="settings_local_server_dialog_title">Dirección del servidor local</string>
    <string name="settings_local_server_dialog_note">Se guarda solo en este dispositivo, sin cifrar: no es un secreto, es una dirección en tu red.</string>
    <string name="settings_local_server_dialog_field_label">Host:puerto</string>
```

```xml
    <string name="notice_local_server_saved">Dirección del servidor local guardada.</string>
```

Match each new string's exact insertion point to its English counterpart
above (i.e. next to `error_missing_key`, `settings_key_dialog_field_label`,
and `notice_key_saved` respectively, in the Spanish file).

- [ ] **Step 6: Compile check**

Run: `cd apps/android && ./gradlew compileDebugKotlin`
Expected: BUILD SUCCESSFUL. Also run
`cd apps/android && ./gradlew lintDebug` if time allows — this catches
missing/mismatched string resources between locales, which a compile
check alone won't (Android string resources aren't type-checked by
Kotlin).

- [ ] **Step 7: Manual verification (device/emulator required)**

If a device/emulator is available (`adb devices` shows one), run:
`cd apps/android && ./gradlew installDebug`, open the app, go to
Settings, and confirm: the "Local server" group appears with an "Add
server address" row; tapping it opens a dialog; entering `192.168.1.20:8000`
and saving shows the address in the row and switches
`vm.conversationProvider` to `LOCAL_SERVER` (observable via the row now
reading "Change server address" and showing the saved address). If no
device/emulator is available, note that in your report — this is a real
environment limitation, not something to fake past.

- [ ] **Step 8: Commit**

```bash
git add apps/android/app/src/main/java/chat/mural/ui/SettingsScreen.kt apps/android/app/src/main/res/values/strings.xml apps/android/app/src/main/res/values-es/strings.xml
git commit -m "Add Local server settings row and dialog"
```

---

## Self-Review Notes

- **Spec coverage:** All 5 corrected file-touch points from the spec's
  "Android client changes" section are covered: `ConversationProvider`
  enum (Task 1), `APIClient` baseUrl/auth (Task 3), settings UI (Task 5,
  in `SettingsScreen.kt` as corrected above, not `AccountSheet.kt`),
  `MuralViewModel`'s provider switch (Task 4), and the `cloudReady()` gate
  (Task 4). The `teaching()`/`/responses` gap the spec didn't mention is
  also covered (Task 4, Step 6).
- **Type consistency checked:** `canStart`'s new parameter order
  (`choice, hasKey, hasLocalServer, hosted`) is used identically in
  Task 1's test file and Task 4's `start()` call site.
  `parseLocalServerUrl`'s `HttpUrl?` return type is consumed identically
  by `localApiClient()` (Task 4) and `saveLocalServerAddress()` (Task 4) —
  both treat `null` as "invalid", never assume non-null.
  `APIClient.local(baseUrl: HttpUrl)` matches exactly how `localApiClient()`
  calls it.
- **No placeholders:** every step has real, complete code or a concrete
  command with an expected result. The two "manual verification" steps
  (Task 2 Step 7, Task 5 Step 7) are conditional on device/emulator
  availability because that's a genuine environment fact for this
  project's existing `androidTest` suite, not something left undecided.
