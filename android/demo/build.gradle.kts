// DviTTS demo app: Jetpack Compose + Material 3. Import or download the model folder once, pick a
// named voice per language, type, Speak / Share. Settings holds the phone bench and the log.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.dvitts.demo"
    compileSdk = 34
    defaultConfig {
        applicationId = "ai.dvitts.demo"
        minSdk = 26
        targetSdk = 34
        versionCode = 4
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    // Release signing: one fixed private key so updates install over older copies. The build box
    // passes the keystore path and password in env vars (loaded from private storage, never in this repo); without
    // them (test builds) the release APK falls back to the debug key.
    val ks = System.getenv("DVITTS_KS")
    signingConfigs {
        if (ks != null) create("release") {
            storeFile = file(ks); storePassword = System.getenv("DVITTS_KS_PASS")
            keyAlias = System.getenv("DVITTS_KS_ALIAS") ?: "dvitts"; keyPassword = System.getenv("DVITTS_KS_PASS")
        }
    }
    buildTypes { release { isMinifyEnabled = false
        signingConfig = if (ks != null) signingConfigs.getByName("release") else signingConfigs.getByName("debug") } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.14" }   // matches Kotlin 1.9.24
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation(project(":android"))
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
