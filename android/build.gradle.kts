// DviTTS on-device library: core = pure Kotlin/JVM port of python/dvitts_onnx.py
// (any JVM; ONNX Runtime is compileOnly), android = AAR wrapper on onnxruntime-android, demo = sample app.
plugins {
    id("org.jetbrains.kotlin.jvm") version "1.9.24" apply false
    id("com.android.library") version "8.5.2" apply false
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
}
