plugins { id("org.jetbrains.kotlin.jvm") }

kotlin { jvmToolchain(17) }

dependencies {
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.32.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.32.0")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5:1.9.24")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "6g"
    testLogging { showStandardStreams = true; events("failed") }
    // DVITTS_MODEL = ONNX model folder, DVITTS_GOLDEN = golden vectors from the Python runner, DVITTS_REPORT = json report path
    listOf("DVITTS_MODEL", "DVITTS_GOLDEN", "DVITTS_REPORT").forEach { k -> System.getenv(k)?.let { environment(k, it) } }
}
