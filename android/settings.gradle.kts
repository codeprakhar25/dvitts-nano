pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "dvitts_android"
include(":core")
if (System.getenv("ANDROID_HOME") != null) include(":android", ":demo")
