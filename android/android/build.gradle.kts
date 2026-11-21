// DviTTS Android library (AAR): the pure-Kotlin core + ONNX Runtime for Android + model download and playback helpers.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.dvitts.android"
    compileSdk = 34
    defaultConfig { minSdk = 26 }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    api(project(":core"))
    api("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
}
