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
