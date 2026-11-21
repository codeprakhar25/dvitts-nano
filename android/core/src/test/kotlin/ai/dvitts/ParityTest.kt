package ai.dvitts

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Kotlin port vs the python spec on golden vectors written by the Python runner: number words (7 languages), front end (normalize,
 * language guess, split), tokenizer ids, and whole say() calls replayed with the python run's random draws.
 * Pass = every text and id identical; audio within float noise (the FFT and exp are not bit-identical to numpy).
 */
class ParityTest {
    private val model = File(System.getenv("DVITTS_MODEL") ?: "model")
    private val golden = File(System.getenv("DVITTS_GOLDEN") ?: "golden")
    private val report = LinkedHashMap<String, Any>()

    /** Replays the python streams in creation order (one per stage-1 draw, one per MeanFlow call). */
    private class Replay(val u: DoubleArray, val n: FloatArray?) : Rng {
        var iu = 0; var inn = 0
        override fun uniform(): Double = u[iu++]
        override fun normal(n: Int): FloatArray { val a = this.n!!.copyOfRange(inn, inn + n); inn += n; return a }
    }

    @Test
    fun parity() {
        val nw = NumWords(Json.parse(File(model, "frontend_numwords.json").readText()).obj())
        val fe = Frontend(nw)
        var ok = true
        // number words
        val nwG = Json.parse(File(golden, "numwords.json").readText()).obj()
        val nwBad = ArrayList<String>(); var nwN = 0
        for ((lang, m) in nwG) for ((k, want) in m.obj()) {
            nwN++
            val got = try { nw.words(k, lang) } catch (e: Exception) { "ERR ${e.message}" }
            if (got != want) nwBad.add("$lang $k: '$got' != '$want'")
        }
        println("KT numwords n=$nwN mismatch=${nwBad.size} ${nwBad.take(8)}")
        report["numwords"] = mapOf("n" to nwN, "mismatch" to nwBad.size, "examples" to nwBad.take(20))
        ok = ok && nwBad.isEmpty()
        // front end
        val fr = Json.parse(File(golden, "frontend.json").readText()).list().map { it.obj() }
        val feBad = ArrayList<String>(); val gBad = ArrayList<String>(); val spBad = ArrayList<String>()
        for (r in fr) {
            val lang = r["lang"] as String; val text = r["text"] as String
            val got = try { fe.normalize(text, lang) } catch (e: Exception) { "ERR ${e.message}" }
            if (got != r["norm"]) feBad.add("[$lang] $text\n   py: ${r["norm"]}\n   kt: $got")
            val g = Script.guessLang(text)
            if (g != r["guess"]) gBad.add("$text: $g != ${r["guess"]}")
            r["split"]?.let { sp ->
                val k = fe.split(r["norm"] as String, lang)
                if (k != sp.list()) spBad.add("[$lang] ${(r["norm"] as String).take(60)}: ${k.size} vs ${sp.list().size} chunks")
            }
        }
        println("KT frontend n=${fr.size} mismatch=${feBad.size} guess_mismatch=${gBad.size} split_mismatch=${spBad.size}")
        feBad.take(12).forEach { println("KT fe $it") }
        report["frontend"] = mapOf("n" to fr.size, "mismatch" to feBad.size, "guess_mismatch" to gBad.size,
                                   "split_mismatch" to spBad.size, "examples" to feBad.take(40), "split_examples" to spBad.take(10))
        ok = ok && feBad.isEmpty() && gBad.isEmpty() && spBad.isEmpty()
        // tokenizer
        val tk = Tokenizer(File(model, "tokenizer.json"))
        val tg = Json.parse(File(golden, "tokens.json").readText()).list().map { it.obj() }
        val tBad = ArrayList<String>()
        val t0 = System.nanoTime()
        for (r in tg) {
            val want = r["ids"].list().map { it.int() }
            val got = tk.encode(r["text"] as String).toList()
            if (got != want) tBad.add("${(r["text"] as String).take(80)}: first diff at ${got.zip(want).indexOfFirst { it.first != it.second }}")
        }
        val tMs = (System.nanoTime() - t0) / 1e6
        println("KT tokens n=${tg.size} mismatch=${tBad.size} ms=${"%.0f".format(tMs)} ${tBad.take(5)}")
        report["tokens"] = mapOf("n" to tg.size, "mismatch" to tBad.size, "ms" to tMs, "examples" to tBad.take(20))
        ok = ok && tBad.isEmpty()
        // runner replays
        val cases = Json.parse(File(golden, "runner/cases.json").readText()).list().map { it.obj() }
        val rows = ArrayList<Map<String, Any>>()
        // Every memory setting must replay the Python goldens exactly.
        for (mode in listOf("off", "all", "pool_off", "pattern_off")) DviTTS(model, threads = 4, memMode = mode).use { tts ->
            for (c in cases) try {
                val z = Npz.read(File(golden, "runner/case${c["k"].int()}.npz"))
                val ns = c["n_streams"].int()
                val streams = ArrayDeque((0 until ns).map { i -> Replay(z.getValue("u$i").doubles(), z["n$i"]?.floats()) })
                tts.rngFactory = { streams.removeFirst() }
                tts.gapRetryS = (c["gap_retry_s"] as? Number)?.toDouble() ?: ((tts.cfg["gap_retry_s"] as? Number)?.toDouble() ?: 0.0)
                val info = ArrayList<ChunkInfo>()
                val t1 = System.nanoTime()
                val y = tts.say(c["text"] as String, c["voice"] as String, seed = c["seed"].int().toLong(), lang = c["lang"] as String, info = info)
                val secs = (System.nanoTime() - t1) / 1e9
                val want = z.getValue("audio").floats()
                val pyChunks = c["chunks"].list().map { it.obj() }
                val idsSame = info.size == pyChunks.size && info.zip(pyChunks).all { (a, b) -> a.ids.toList() == b["ids"].list().map { it.int() } } &&
                    info.zip(pyChunks).all { (a, b) -> b["gap_retry"] == null || a.gapRetry == b["gap_retry"].int() }
                // python records a chunk's text only on the long-text path (single calls keep it implicit)
                val textSame = info.zip(pyChunks).all { (a, b) -> b["text"] == null || a.text == b["text"] }
                var md = 0.0
                for (i in 0 until minOf(y.size, want.size)) md = maxOf(md, Math.abs((y[i] - want[i]).toDouble()))
                val row = mapOf("mode" to mode, "k" to c["k"].int(), "lang" to c["lang"] as String, "ids_same" to idsSame, "chunk_text_same" to textSame,
                                "len_kt" to y.size, "len_py" to want.size, "max_diff" to md, "rtf" to secs / (y.size / 24000.0))
                println("KT runner $row")
                rows.add(row)
                ok = ok && idsSame && textSame && y.size == want.size && md < 1e-3
            } catch (e: Exception) {                              // e.g. more random draws than Python: ids diverged
                println("KT runner {mode=$mode, k=${c["k"].int()}, error=${e.javaClass.simpleName}}")
                rows.add(mapOf("mode" to mode, "k" to c["k"].int(), "error" to e.javaClass.simpleName))
                ok = false
            }
        }
        report["runner"] = rows
        System.getenv("DVITTS_REPORT")?.let { File(it).writeText(toJson(report)) }
        println("KT summary " + (if (ok) "PASS" else "CHECK"))
        assertTrue(ok, "parity mismatches: see KT lines")
    }

    private fun toJson(v: Any?): String = when (v) {
        null -> "null"
        is String -> "\"" + v.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""
        is Number, is Boolean -> v.toString()
        is Map<*, *> -> v.entries.joinToString(",", "{", "}") { "${toJson(it.key.toString())}:${toJson(it.value)}" }
        is List<*> -> v.joinToString(",", "[", "]") { toJson(it) }
        else -> toJson(v.toString())
    }
}
