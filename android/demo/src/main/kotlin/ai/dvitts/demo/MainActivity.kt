@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package ai.dvitts.demo

import ai.dvitts.ChunkInfo
import ai.dvitts.DviTTS
import ai.dvitts.Json
import ai.dvitts.android.AudioOut
import ai.dvitts.android.ModelStore
import ai.dvitts.obj
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** One preset as the app shows it (voices/index.json; a folder without names falls back to the clip id). */
data class VoiceUi(val id: String, val name: String, val lang: String, val gender: String, val source: String)

val LANGS = listOf("en", "hi", "mr", "gu", "bn", "pa", "or")
val LANG_NATIVE = mapOf("en" to "English", "hi" to "हिन्दी", "mr" to "मराठी", "gu" to "ગુજરાતી", "bn" to "বাংলা", "pa" to "ਪੰਜਾਬੀ", "or" to "ଓଡ଼ିଆ")
val LANG_EN = mapOf("en" to "English", "hi" to "Hindi", "mr" to "Marathi", "gu" to "Gujarati", "bn" to "Bengali", "pa" to "Punjabi", "or" to "Odia")
val LANG_TINT = mapOf("en" to Color(0xFF3F51B5), "hi" to Color(0xFFE8711A), "mr" to Color(0xFFC2185B), "gu" to Color(0xFF00897B),
    "bn" to Color(0xFF7B1FA2), "pa" to Color(0xFFF9A825), "or" to Color(0xFF2E7D32))

val SAMPLES = mapOf(
    "en" to listOf("Hello! The weather is lovely today, so let's take a walk in the park this evening.",
                   "Your order has been shipped and will reach you by Friday afternoon."),
    "hi" to listOf("नमस्ते! आज मौसम बहुत अच्छा है, चलिए शाम को पार्क में टहलने चलते हैं।",
                   "मेरी ट्रेन सुबह 6:15 पर है, इसलिए आज मैं जल्दी सो जाऊँगा।"),
    "mr" to listOf("नमस्कार! आज हवामान खूप छान आहे."),
    "gu" to listOf("નમસ્તે! આજે હવામાન ખૂબ સરસ છે."),
    "bn" to listOf("নমস্কার! আজ আবহাওয়া খুব সুন্দর।"),
    "pa" to listOf("ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।"),
    "or" to listOf("ନମସ୍କାର! ଆଜି ପାଣିପାଗ ବହୁତ ଭଲ।"))

private val Saffron = Color(0xFFE8711A)
private val Indigo = Color(0xFF3949AB)
private val LightColors = lightColorScheme(primary = Saffron, onPrimary = Color.White, primaryContainer = Color(0xFFFFE0C7),
    onPrimaryContainer = Color(0xFF4A1F00), secondary = Indigo, onSecondary = Color.White, secondaryContainer = Color(0xFFDEE1FF),
    onSecondaryContainer = Color(0xFF0E1A5C), background = Color(0xFFFFFBF7), surface = Color(0xFFFFFBF7),
    surfaceVariant = Color(0xFFF4EDE6), onSurfaceVariant = Color(0xFF52463C))
private val DarkColors = darkColorScheme(primary = Color(0xFFFFB68A), onPrimary = Color(0xFF552200), primaryContainer = Color(0xFF7A3400),
    onPrimaryContainer = Color(0xFFFFDBC8), secondary = Color(0xFFBAC3FF), onSecondary = Color(0xFF1A2A80),
    secondaryContainer = Color(0xFF2F3D96), onSecondaryContainer = Color(0xFFDEE1FF), background = Color(0xFF17120E),
    surface = Color(0xFF17120E), surfaceVariant = Color(0xFF2A221C), onSurfaceVariant = Color(0xFFD7C3B6))

class MainActivity : ComponentActivity() {
    private var tts: DviTTS? = null
    private var loadedThreads = 0
    private var loadedMb = -1L
    private val log = StringBuilder()
    private var logState by mutableStateOf("")

    // ONNX Runtime memory setting: pool + pattern off ("all"): peak 1125 -> 865 MB on a Snapdragon 8 Gen 3 phone, same
    // speed, identical audio.
    private fun memMode(): String = "all"

    private fun note(s: String) { synchronized(log) { log.append(s).append('\n'); logState = log.toString() } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        note("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, ${soc()}, ${Runtime.getRuntime().availableProcessors()} cores")
        note("Memory setting: ${memMode()}  (app start: memory ${mem("VmRSS")} MB)")
        setContent {
            val colors: ColorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors
            MaterialTheme(colorScheme = colors) { App() }
        }
    }

    override fun onDestroy() { AudioOut.stop(); tts?.close(); super.onDestroy() }

    // ---- state --------------------------------------------------------------------------------------------------
    private fun readVoices(): List<VoiceUi> {
        val f = File(ModelStore.dir(this), "voices/index.json")
        if (!f.exists()) return emptyList()
        val idx = Json.parse(f.readText()).obj()
        return idx.keys.map { id ->
            val m = idx[id].obj()
            val g = (m["gender"] as? String) ?: when { id.contains("Female") -> "female"; id.contains("Male") -> "male"; else -> "" }
            VoiceUi(id, (m["name"] as? String) ?: id, m["lang"] as String, g, (m["source"] as? String) ?: "")
        }.sortedWith(compareBy({ LANGS.indexOf(it.lang) }, { it.gender != "female" }, { it.name }))
    }

    private fun engine(threads: Int): DviTTS {
        if (tts == null || loadedThreads != threads) {
            tts?.close()
            val t0 = System.nanoTime()
            tts = ModelStore.open(this, threads, memMode()); loadedThreads = threads
            loadedMb = mem("VmRSS")
            note("Loaded in %.1f s (%d threads, memory setting %s), memory %d MB".format((System.nanoTime() - t0) / 1e9, threads, memMode(), loadedMb))
        }
        return tts!!
    }

    // ---- UI -----------------------------------------------------------------------------------------------------
    @Composable
    fun App() {
        var ready by remember { mutableStateOf(ModelStore.ready(this)) }
        var voices by remember { mutableStateOf(if (ready) readVoices() else emptyList()) }
        var settings by remember { mutableStateOf(false) }
        var threads by remember { mutableFloatStateOf(4f) }
        if (settings) {
            Settings(threads = threads, onThreads = { threads = it }, onBack = { settings = false },
                onReset = { tts?.close(); tts = null; ModelStore.dir(this).deleteRecursively(); ready = false; voices = emptyList(); settings = false },
                voices = voices)
            return
        }
        Scaffold(topBar = {
            TopAppBar(title = {
                Column {
                    Text("DviTTS", fontWeight = FontWeight.Bold)
                    Text("Indian voices, on your phone", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }, actions = { IconButton(onClick = { settings = true }) { Icon(Icons.Filled.Settings, "Settings") } },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background))
        }) { pad ->
            Box(Modifier.padding(pad).fillMaxSize()) {
                if (!ready) Setup(onReady = { ready = true; voices = readVoices() })
                else Studio(voices, threads.toInt())
            }
        }
    }

    @Composable
    fun Setup(onReady: () -> Unit) {
        val scope = rememberCoroutineScope()
        var link by remember { mutableStateOf("") }
        var pct by remember { mutableStateOf(-1) }
        var msg by remember { mutableStateOf("") }
        var busy by remember { mutableStateOf(false) }
        fun start(work: suspend ((Int) -> Unit) -> Unit) {
            busy = true; msg = ""; pct = 0
            scope.launch {
                try {
                    withContext(Dispatchers.IO) { work { p -> pct = p } }
                    note("Model ready: ${ModelStore.name(this@MainActivity)}, every file passed its md5 check.")
                    onReady()
                } catch (e: Throwable) {
                    msg = "Could not set up the model: ${e.message}"; note("ERROR ${e.javaClass.simpleName}: ${e.message}")
                } finally { busy = false }
            }
        }
        val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) start { p ->
                val size = contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
                    if (c.moveToFirst()) c.getLong(0) else -1L } ?: -1L
                contentResolver.openInputStream(uri)!!.use { ModelStore.unpack(this@MainActivity, it, size, p) }
            }
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Set up the voice model", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            Text("DviTTS runs fully on this phone, with no internet after setup. It needs its model files once (about 350 MB).",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Import the model file", fontWeight = FontWeight.SemiBold)
                    Text("Got the .zip on this phone (Drive, WhatsApp, USB)? Pick it here. Fastest.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = { picker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream")) },
                        enabled = !busy) { Text("Choose .zip file") }
                }
            }
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Or download from a link", fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(link, { link = it.trim() }, Modifier.fillMaxWidth(), label = { Text("https://… .zip") }, singleLine = true)
                    FilledTonalButton(onClick = { start { p -> ModelStore.download(this@MainActivity, link, p) } },
                        enabled = !busy && link.startsWith("http")) { Text("Download") }
                }
            }
            if (busy) {
                LinearProgressIndicator(progress = { (pct.coerceAtLeast(0)) / 100f }, modifier = Modifier.fillMaxWidth())
                Text(if (pct >= 0) "Setting up… $pct %" else "Setting up…", style = MaterialTheme.typography.bodySmall)
            }
            if (msg.isNotEmpty()) Text(msg, color = MaterialTheme.colorScheme.error)
        }
    }

    @Composable
    fun Studio(voices: List<VoiceUi>, threads: Int) {
        val scope = rememberCoroutineScope()
        var lang by remember { mutableStateOf("hi") }
        val shown = voices.filter { it.lang == lang }
        var sel by remember { mutableStateOf(voices.firstOrNull { it.lang == "hi" }?.id ?: voices.firstOrNull()?.id ?: "") }
        var text by remember { mutableStateOf(SAMPLES["hi"]!![0]) }
        var busy by remember { mutableStateOf(false) }
        var status by remember { mutableStateOf("") }
        var switched by remember { mutableStateOf("") }
        var last by remember { mutableStateOf<FloatArray?>(null) }
        var playing by remember { mutableStateOf(false) }
        var played by remember { mutableFloatStateOf(0f) }
        val byId = voices.associateBy { it.id }

        LaunchedEffect(playing, last) {
            val y = last
            while (playing && y != null) {
                played = (AudioOut.position().toFloat() / y.size).coerceIn(0f, 1f)
                if (AudioOut.position() >= y.size) { playing = false; played = 0f }
                delay(50)
            }
        }

        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(LANGS.filter { l -> voices.any { it.lang == l } }) { l ->
                    FilterChip(selected = l == lang, onClick = {
                        if (text.isBlank() || SAMPLES.values.flatten().contains(text)) text = SAMPLES[l]?.first() ?: text
                        lang = l; sel = voices.firstOrNull { it.lang == l }?.id ?: sel
                    }, label = { Text(LANG_NATIVE[l] ?: l) })
                }
            }
            Text("Voice", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), contentPadding = PaddingValues(end = 8.dp)) {
                items(shown) { v -> VoiceCard(v, v.id == sel) { sel = v.id } }
            }
            OutlinedTextField(text, { text = it }, Modifier.fillMaxWidth().heightIn(min = 140.dp),
                label = { Text("Type anything to say") }, textStyle = MaterialTheme.typography.bodyLarge.copy(fontSize = 18.sp))
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(SAMPLES[lang].orEmpty()) { s ->
                    AssistChip(onClick = { text = s }, label = { Text(s, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.width(180.dp)) })
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    if (playing) { AudioOut.stop(); playing = false; return@Button }
                    busy = true; status = ""; switched = ""
                    scope.launch {
                        try {
                            val (y, msg, sw) = withContext(Dispatchers.Default) {
                                val t = engine(threads)
                                val info = ArrayList<ChunkInfo>()
                                val t0 = System.nanoTime()
                                val y = t.say(text, sel, seed = System.currentTimeMillis() % 100000, info = info)
                                val s = (System.nanoTime() - t0) / 1e9
                                val a = y.size / 24000.0
                                note("say %.1f s audio in %.2f s (%.2fx real time), %d chunk(s)".format(a, s, s / a, info.size))
                                val sw = if (t.lastXling) {
                                    val n = byId[sel]?.name ?: sel
                                    "$n kept; S1 prompt dropped (cross-language)."
                                } else ""
                                Triple(y, "%.1f s of speech, made in %.1f s on this phone".format(a, s), sw)
                            }
                            last = y; status = msg; switched = sw
                            AudioOut.play(y); played = 0f; playing = true
                        } catch (e: Throwable) {
                            status = "Could not speak: ${e.message}"; note("ERROR ${e.javaClass.simpleName}: ${e.message}")
                        } finally { busy = false }
                    }
                }, enabled = !busy && text.isNotBlank() && sel.isNotEmpty(), modifier = Modifier.height(56.dp).weight(1f),
                    shape = RoundedCornerShape(28.dp)) {
                    if (busy) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.5.dp, color = MaterialTheme.colorScheme.onPrimary)
                        Spacer(Modifier.width(10.dp)); Text("Making speech…")
                    } else if (playing) {
                        Icon(Icons.Filled.Close, null); Spacer(Modifier.width(8.dp)); Text("Stop")
                    } else {
                        Icon(Icons.Filled.PlayArrow, null); Spacer(Modifier.width(8.dp)); Text("Speak", fontSize = 18.sp)
                    }
                }
                OutlinedButton(onClick = { last?.let { share(it) } }, enabled = last != null && !busy, modifier = Modifier.height(56.dp),
                    shape = RoundedCornerShape(28.dp)) { Icon(Icons.Filled.Share, "Share") }
            }
            AnimatedVisibility(playing) { LinearProgressIndicator(progress = { played }, modifier = Modifier.fillMaxWidth()) }
            if (status.isNotEmpty()) Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (switched.isNotEmpty()) Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
                Text(switched, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
            }
            Spacer(Modifier.height(8.dp))
            Text("Runs offline. Every clip carries an inaudible AudioSeal watermark. Do not use a voice to pass as a real person.",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable
    fun VoiceCard(v: VoiceUi, selected: Boolean, onClick: () -> Unit) {
        val tint = LANG_TINT[v.lang] ?: MaterialTheme.colorScheme.primary
        Card(onClick = onClick, modifier = Modifier.width(132.dp),
            border = if (selected) BorderStroke(2.dp, MaterialTheme.colorScheme.primary) else null,
            colors = CardDefaults.cardColors(containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant)) {
            Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Box(Modifier.size(52.dp).clip(CircleShape).background(tint), contentAlignment = Alignment.Center) {
                    Text(v.name.take(1).uppercase(), color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                }
                Text(v.name, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("${LANG_EN[v.lang] ?: v.lang} · ${v.gender}", style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
            }
        }
    }

    @Composable
    fun Settings(threads: Float, onThreads: (Float) -> Unit, onBack: () -> Unit, onReset: () -> Unit, voices: List<VoiceUi>) {
        val scope = rememberCoroutineScope()
        var benching by remember { mutableStateOf(false) }
        Scaffold(topBar = {
            TopAppBar(title = { Text("Settings") }, navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, "Back") } })
        }) { pad ->
            Column(Modifier.padding(pad).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("Model: ${ModelStore.name(this@MainActivity) ?: "not installed"}", style = MaterialTheme.typography.bodyMedium)
                Text("CPU threads: ${threads.toInt()}", style = MaterialTheme.typography.labelLarge)
                Slider(threads, onThreads, valueRange = 1f..8f, steps = 6)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FilledTonalButton(onClick = {
                        benching = true
                        scope.launch { try { withContext(Dispatchers.Default) { runBench(threads.toInt(), voices) } } catch (e: Throwable) {
                            note("ERROR ${e.javaClass.simpleName}: ${e.message}") } finally { benching = false } }
                    }, enabled = !benching && voices.isNotEmpty()) { Text(if (benching) "Running…" else "Phone bench") }
                    OutlinedButton(onClick = {
                        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        cm.setPrimaryClip(ClipData.newPlainText("dvitts", logState))
                    }) { Text("Copy log") }
                }
                Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
                    SelectionContainer { Text(logState.ifEmpty { "Log is empty." }, Modifier.padding(12.dp), fontFamily = FontFamily.Monospace, fontSize = 11.sp) }
                }
                HorizontalDivider()
                Text("Voices", style = MaterialTheme.typography.titleSmall)
                Text(voices.map { it.source }.filter { it.isNotEmpty() }.distinct().joinToString(" · ").ifEmpty { "-" },
                    style = MaterialTheme.typography.bodySmall)
                Text("Voice names are made up. The recordings come from the open datasets above, credited under CC BY 4.0.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                HorizontalDivider()
                TextButton(onClick = onReset, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) {
                    Text("Remove model files")
                }
            }
        }
    }

    // ---- actions ------------------------------------------------------------------------------------------------
    private fun share(y: FloatArray) {
        val d = File(cacheDir, "share").apply { mkdirs() }
        val f = File(d, "dvitts.wav").apply { writeBytes(AudioOut.wav(y)) }
        val uri = FileProvider.getUriForFile(this, "$packageName.files", f)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share clip"))
    }

    /** Phone test: 1 warm-up + 8 timed sentences (hi + en), watermark on. Prints the DEMO_SUMMARY line. */
    private fun runBench(threads: Int, voices: List<VoiceUi>) {
        val t = engine(threads)
        val hiV = voices.firstOrNull { it.lang == "hi" }?.id ?: return note("No Hindi voice.")
        val enV = voices.firstOrNull { it.lang == "en" }?.id ?: voices.first().id
        val items = listOf(
            hiV to "आज का दिन बहुत अच्छा है।",
            hiV to "मैंने 2024 में 3,500 रुपये दिए और बाकी अगले महीने दूँगा।",
            hiV to "रेलगाड़ी सुबह 06:15 पर निकलेगी, इसलिए हमें समय पर स्टेशन पहुँचना होगा।",
            hiV to "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ बोली जाती हैं और हर राज्य की अपनी परंपराएँ हैं।",
            hiV to "कृपया अपना फ़ोन चार्ज कर लीजिए।",
            enV to "Please charge your phone before the trip.",
            enV to "The meeting starts at 10:30, so please be on time.",
            enV to "I paid 45 dollars for the tickets and 12 for the snacks.",
            enV to "The village library opened at nine every morning, and by ten the reading room was usually full.")
        var aud = 0.0; var wall = 0.0; var noEos = 0; var redraws = 0; val firsts = ArrayList<Double>()
        for ((i, p) in items.withIndex()) {
            val info = ArrayList<ChunkInfo>()
            val t0 = System.nanoTime()
            val y = t.say(p.second, p.first, seed = i.toLong(), info = info)
            val s = (System.nanoTime() - t0) / 1e9
            val a = y.size / 24000.0
            note("%d %s %.1fs audio  %.2fs  rtf %.3f".format(i, if (i == 0) "warm-up" else "", a, s, s / a))
            if (i == 0) continue
            aud += a; wall += s; firsts.add(s)
            noEos += info.count { !it.eos }; redraws += info.sumOf { it.redraw }
        }
        note("DEMO_SUMMARY {\"threads\":$threads,\"n\":${items.size - 1},\"audio_s\":%.1f,\"rtf\":%.3f,\"first_audio_s_mean\":%.2f,\"no_eos\":$noEos,\"redraws\":$redraws,\"mem_mode\":\"%s\",\"loaded_mb\":%d,\"rss_mb\":%d,\"rss_peak_mb\":%d,\"model\":\"%s\",\"device\":\"%s\"}"
            .format(aud, wall / aud, firsts.average(), memMode(), loadedMb, mem("VmRSS"), mem("VmHWM"), ModelStore.name(this) ?: "", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} | ${soc()}"))
    }

    private fun mem(key: String): Long = try {
        File("/proc/self/status").readLines().first { it.startsWith(key) }.split(Regex("\\s+"))[1].toLong() / 1024
    } catch (e: Exception) { -1 }

    private fun soc(): String =
        if (android.os.Build.VERSION.SDK_INT >= 31) "${android.os.Build.SOC_MANUFACTURER} ${android.os.Build.SOC_MODEL}" else android.os.Build.HARDWARE
}
