package ai.dvitts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.SplittableRandom

/** Random draws for one stream: stage 1 takes uniform() per sampled id, MeanFlow takes normal(n) once. */
interface Rng {
    fun uniform(): Double
    fun normal(n: Int): FloatArray
}

/** Default generator (any good one works; the python runner uses numpy PCG64, so audio differs per port, not in kind). */
class DefaultRng(seed: Long) : Rng {
    private val r = SplittableRandom(seed)
    override fun uniform(): Double = r.nextDouble()
    override fun normal(n: Int): FloatArray = FloatArray(n) {
        var u1 = r.nextDouble(); while (u1 <= 0.0) u1 = r.nextDouble()
        (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math.PI * r.nextDouble())).toFloat()
    }
}

class Voice(val name: String, val refTextIds: LongArray, val refIds: LongArray, val cR: LongArray, val acR: LongArray)

/** Per-call details (chunk texts, ids, stop flags, redraws). */
class ChunkInfo(val text: String) {
    var ids: IntArray = IntArray(0); var eos = false; var redraw = 0; var capIds = 0; var minRatio = 0.0
    var gap = 0.0; var gapRetry = 0; var gapFirst = 0.0          // gap retry, see sayOne
}

/**
 * DviTTS on ONNX Runtime: Kotlin port of dvitts_onnx.py (the spec). One model folder, any JVM / Android.
 *   val tts = DviTTS(File("dvitts-onnx"))
 *   val audio = tts.say("नमस्ते, आप कैसे हैं?", "rasa_hi_Female__c012")   // FloatArray, 24 kHz mono
 */
class DviTTS(val dir: File, threads: Int = 4, val watermark: Boolean = true, val trim: Boolean = true,
             var rngFactory: (Long) -> Rng = { DefaultRng(it) }, val memMode: String = "all") : AutoCloseable {
    val cfg: Map<String, Any?> = Json.parse(File(dir, "config.json").readText()).obj()
    /** Gap retry trigger in seconds (config gap_retry_s; absent in older folders = off). Settable for tests. */
    var gapRetryS: Double = (cfg["gap_retry_s"] as? Number)?.toDouble() ?: 0.0
    val sr: Int = cfg["sr"].int()
    private val env = OrtEnvironment.getEnvironment()
    private val s1: OrtSession; private val mf: OrtSession; private val dec: OrtSession
    private val wm: OrtSession?; private val vad: OrtSession?
    val tokenizer = Tokenizer(File(dir, "tokenizer.json"))
    val numWords = NumWords(Json.parse(File(dir, "frontend_numwords.json").readText()).obj())
    val frontend = Frontend(numWords)
    private val tmNew: HashMap<Int, Int> = HashMap()
    private val tmFb: HashMap<Int, IntArray> = HashMap()
    private val vals: FloatArray; private val basis: LongArray
    private val rsKernel: FloatArray; private val rsWidth: Int; private val rsK: Int
    private val instr: IntArray = cfg["instruct_ids"].list().map { it.int() }.toIntArray()
    private val voices = HashMap<String, Voice>()

    init {
        val f = cfg["files"].obj()
        fun opts(n: Int) = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(n); setInterOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")     // off: spinning stole cores on phones
            // Memory, memMode: "off" = ONNX Runtime defaults, "all" = pool (arena) + memory pattern off (default),
            // "pool_off", "pattern_off". On a Snapdragon 8 Gen 3 phone "all" cuts peak memory 1125 -> 865 MB at the same
            // speed, with identical audio. (The Java API has no shared-pool call.)
            when (memMode) {
                "off" -> {}
                "all" -> { setCPUArenaAllocator(false); setMemoryPatternOptimization(false) }
                "pool_off" -> setCPUArenaAllocator(false)
                "pattern_off" -> setMemoryPatternOptimization(false)
                else -> throw IllegalArgumentException("memMode: $memMode")
            }
        }
        val o = opts(threads)
        fun sess(key: String, so: OrtSession.SessionOptions) = env.createSession(File(dir, f[key] as String).path, so)
        s1 = sess("s1", o); mf = sess("mf", o); dec = sess("dec", o)
        wm = if (watermark) sess("wm", o) else null
        vad = if (trim) sess("vad", opts(1)) else null
        val tm = Json.parse(File(dir, "text_map.json").readText()).obj()
        tm["keep"].list().forEachIndexed { n, old -> tmNew[old.int()] = n }
        for ((k, v) in tm["fb"].obj()) tmFb[k.toInt()] = v.list().map { tmNew.getValue(it.int()) }.toIntArray()
        val c = Npz.read(File(dir, "consts.npz"))
        vals = c.getValue("vals").floats(); basis = c.getValue("basis").longs()
        val k = c.getValue("rs_kernel")
        rsKernel = k.floats(); rsK = k.shape.last(); rsWidth = c.getValue("rs_width").longs()[0].toInt()
    }

    override fun close() { listOfNotNull(s1, mf, dec, wm, vad).forEach { it.close() } }

    // ---- voices --------------------------------------------------------------------------------------------------
    fun voices(): List<String> = File(dir, "voices").listFiles()!!.filter { it.name.endsWith(".npz") }.map { it.name.removeSuffix(".npz") }.sorted()

    /** voices/index.json: id -> {lang, gender, transcript, source; V1 also name, label}. Empty if the file is missing. */
    val index: Map<String, Map<String, Any?>> by lazy {
        val f = File(dir, "voices/index.json")
        if (f.exists()) Json.parse(f.readText()).obj().mapValues { it.value.obj() } else emptyMap()
    }

    /** Same rule as dvitts_onnx.match_voice: a preset speaks its own side of the English / Indian split only. If the text's
     *  language is on the other side, the first preset of that language with the same gender (Hindi if none), else [name]. */
    fun matchVoice(name: String, lang: String): String {
        val m = index[name] ?: return name
        if ((lang == "en") == (m["lang"] == "en")) return name
        val ids = index.keys.sorted()
        val pool = ids.filter { index[it]!!["lang"] == lang }.ifEmpty { ids.filter { index[it]!!["lang"] == "hi" } }
        val same = pool.filter { index[it]!!["gender"] == m["gender"] }
        return same.firstOrNull() ?: pool.firstOrNull() ?: name
    }

    /** Set by every say(): (asked, used) when matchVoice swapped the preset, else null.
     *  Unused: a mismatch keeps the pick and drops the S1 prompt ([lastXling]). */
    var lastVoiceSwitch: Pair<String, String>? = null
        private set

    /** True when this say() dropped the S1 prompt for an English/Indian mismatch (MeanFlow still uses the pick). */
    var lastXling: Boolean = false
        private set

    fun voice(name: String): Voice = voices.getOrPut(name) {
        val z = Npz.read(File(dir, "voices/$name.npz"))
        Voice(name, z.getValue("ref_text_ids").longs(), z.getValue("ref_ids").longs(), z.getValue("c_r").longs(), z.getValue("ac_r").longs())
    }

    // ---- text ----------------------------------------------------------------------------------------------------
    fun encode(text: String): IntArray = tokenizer.encode(text)

    private fun textMap(ids: IntArray): IntArray {
        val out = ArrayList<Int>(ids.size)
        for (i in ids) { val n = tmNew[i]; if (n != null) out.add(n) else tmFb.getValue(i).forEach { out.add(it) } }
        return out.toIntArray()
    }

    fun expIds(text: String): Double {
        val ts = cfg["tok_s"].obj()
        return 25 * Script.runs(text).sumOf { (n, r) -> encode(" $r").size / (ts[n]?.num() ?: 2.8) }
    }

    fun minRatio(text: String, nText: Int): Double {
        val sc = cfg["min_scale"].obj(); val base = cfg["min_ratio"].num()
        val rt = Script.runs(text).map { (n, r) -> n to encode(" $r").size }
        if (rt.all { (s, _) -> (sc[s]?.num() ?: 1.0) >= 1.0 } || nText <= 0) return base
        val eff = rt.sumOf { (s, n) -> n * minOf(1.0, sc[s]?.num() ?: 1.0) }
        return base * minOf(1.0, eff / nText)
    }

    // ---- public --------------------------------------------------------------------------------------------------
    /** Text -> 24 kHz mono audio. Long text is split into chunks of <= ~25 s, joined with 0.2 s of silence. */
    fun say(text0: String, voice: String, seed: Long = 0, lang: String? = null, normalize: Boolean = true,
            info: MutableList<ChunkInfo>? = null, match: Boolean = true): FloatArray {
        val bad = Script.unsupported(text0)
        require(bad.isEmpty()) { "Unsupported script: ${bad.joinToString(", ")}. This model speaks ${Script.SPOKEN}." }
        val lg = lang ?: Script.guessLang(text0)
        var xling = false
        lastVoiceSwitch = null
        if (match && matchVoice(voice, lg) != voice) xling = true
        lastXling = xling
        val v = voice(voice)
        val text = if (normalize) frontend.normalize(text0, lg) else text0
        val splitS = cfg["split_s"].num(); val rate = cfg["rate_scale"].num()
        val inf = info ?: ArrayList()
        var y: FloatArray
        val ic = if (lg != "en") (cfg["in_chunk"] as? Number)?.toInt() ?: 0 else 0   // V1: Indian text in chunks of <= in_chunk chars
        if (text.cpLen() > (Frontend.MAX_CHARS[lg] ?: Frontend.MAX_CHARS_IN) || rate * expIds(text) / 25 > splitS ||
            (ic > 0 && text.cpLen() > ic)) {
            val chunks = ArrayList<String>()
            val first = if (ic > 0) frontend.split(text, lg, maxChars = ic, minChars = (ic * .6).toInt()) else frontend.split(text, lg)
            for (ch in first) {
                val e = rate * expIds(ch) / 25
                if (e <= splitS) { chunks.add(ch); continue }
                val mx = maxOf(40, (ch.cpLen() / Math.ceil(e / splitS)).toInt())
                chunks.addAll(frontend.split(ch, lg, maxChars = mx, minChars = (mx * .6).toInt()))
            }
            val gap = FloatArray((cfg["gap_s"].num() * sr).toInt())
            val parts = ArrayList<FloatArray>()
            for ((i, ch) in chunks.withIndex()) {
                val ci = ChunkInfo(ch); inf.add(ci)
                parts.add(sayOne(v, ch, seed * 1000 + i, ci, xling))
                if (i < chunks.size - 1) parts.add(gap)
            }
            y = concat(parts)
        } else {
            val ci = ChunkInfo(text); inf.add(ci)
            y = sayOne(v, text, seed, ci, xling)
        }
        if (wm != null) y = mark(y)
        return y
    }

    /** One chunk. With trim on, a take whose longest non-speech stretch inside the clip is >= gap_retry_s is made once
     *  more on seed + gap_seed; the take with the shorter inside gap is kept (tie: first). Same as dvitts_onnx.py. */
    private fun sayOne(v: Voice, text: String, seed: Long, info: ChunkInfo, xling: Boolean = false,
                       gapRetry: Boolean = true): FloatArray {
        val tt = encode(" $text")
        val exp = expIds(text)
        val cap = minOf((cfg["cap_x"].num() * exp).toInt(), (cfg["max_s"].num() * 25).toInt())
        val mr = minRatio(text, tt.size)
        val refText = if (xling) IntArray(0) else v.refTextIds.map { it.toInt() }.toIntArray()
        val refIds = if (xling) LongArray(0) else v.refIds
        val prefix = textMap(instr + refText + tt)
        var (ids, eos) = stage1(prefix, refIds, tt.size, rngFactory(seed), mr, cap)
        val redraws = cfg["redraws"].int(); val rseed = (cfg["redraw_seed"] as Number).toLong()
        for (r in 1..redraws) {                                  // runaway guard: draw again if it never stopped
            if (eos) break
            val (i2, e2) = stage1(prefix, refIds, tt.size, rngFactory(seed + r * rseed), mr, cap)
            info.redraw = r
            if (e2) { ids = i2; eos = e2 }
        }
        info.ids = ids; info.eos = eos; info.capIds = cap; info.minRatio = mr
        val n24 = ids.size * (sr / 25)
        if (ids.size < 5) return FloatArray(n24)
        val aHat = meanflow(v, ids, rngFactory(seed))
        var y = decode(ids, aHat, n24)
        if (vad == null) return y
        val b = bounds(y)
        val pad = cfg["trim_pad"].num()
        val a0 = maxOf(0, ((b[1] - pad) * sr).toInt()); val a1 = minOf(y.size, ((b[0] - b[2] + pad) * sr).toInt())
        y = if (a1 > a0) y.copyOfRange(a0, a1) else FloatArray(0)
        info.gap = b[3]; info.gapRetry = 0
        if (gapRetry && gapRetryS > 0 && b[3] >= gapRetryS) {
            val i2 = ChunkInfo(text)
            val y2 = sayOne(v, text, seed + (cfg["gap_seed"] as Number).toLong(), i2, xling, gapRetry = false)
            if (i2.gap < b[3]) {
                info.ids = i2.ids; info.eos = i2.eos; info.redraw = i2.redraw; info.capIds = i2.capIds
                info.minRatio = i2.minRatio; info.gap = i2.gap
                y = y2
            }
            info.gapRetry = 1; info.gapFirst = b[3]
        }
        return y
    }

    // ---- stage 1 -------------------------------------------------------------------------------------------------
    fun stage1(textIds: IntArray, refIds: LongArray, nText: Int, rng: Rng, minRatio: Double, cap: Int): Pair<IntArray, Boolean> {
        val off = cfg["offset"].int(); val nsp = cfg["n_speech"].int(); val eosId = cfg["eos"].int()
        val sh = cfg["s1_shape"].obj(); val L = sh["L"].int().toLong(); val KV = sh["KV"].int().toLong(); val hd = sh["hd"].int().toLong()
        val prefix = LongArray(textIds.size + 2 + refIds.size)
        prefix[0] = (off + cfg["sos"].int()).toLong()
        for (i in textIds.indices) prefix[1 + i] = textIds[i].toLong()
        prefix[1 + textIds.size] = (off + cfg["task"].int()).toLong()
        for (i in refIds.indices) prefix[2 + textIds.size + i] = off + refIds[i]
        val lo = (nText * minRatio).toInt()
        val hi = minOf(maxOf((nText * cfg["max_ratio"].num()).toInt(), 1), maxOf(cap, lo + 1))
        var pk = OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(L, KV, 0, hd))
        var pv = OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(L, KV, 0, hd))
        var res: OrtSession.Result? = null
        var feed = prefix
        val out = ArrayList<Int>()
        val nrows = cfg["nrows"].int()
        val z = FloatArray(nrows)
        try {
            for (t in 0 until hi) {
                val r = OnnxTensor.createTensor(env, LongBuffer.wrap(feed), longArrayOf(1, feed.size.toLong())).use { ti ->
                    s1.run(mapOf("ids" to ti, "past_k" to pk, "past_v" to pv))
                }
                res?.close() ?: run { pk.close(); pv.close() }
                res = r
                pk = r[1] as OnnxTensor; pv = r[2] as OnnxTensor
                val fb = (r[0] as OnnxTensor).floatBuffer
                fb.position((feed.size - 1) * nrows); fb.get(z)
                for (i in nsp + 2 until nrows) z[i] = Float.NEGATIVE_INFINITY
                z[nsp] = Float.NEGATIVE_INFINITY
                if (t < lo) z[eosId] = Float.NEGATIVE_INFINITY
                val j = sample(z, out, rng)
                if (j == eosId) return out.toIntArray() to true
                out.add(j)
                feed = longArrayOf((off + j).toLong())
            }
        } finally { res?.close() }
        return out.toIntArray() to false
    }

    /** RAS sampler, = dvitts_onnx.DviTTS.sample. */
    fun sample(logits: FloatArray, recent: List<Int>, rng: Rng): Int {
        val s = cfg["sampler"].obj()
        val temp = s["temp"].num(); val topK = s["top_k"].int(); val topP = s["top_p"].num()
        val win = s["win"].int(); val tauR = s["tau_r"].num()
        val v = logits.size
        val p = DoubleArray(v) { logits[it].toDouble() / temp }
        var mx = Double.NEGATIVE_INFINITY
        for (x in p) if (x > mx) mx = x
        for (i in 0 until v) p[i] = Math.exp(p[i] - mx)
        val sum = pairwiseSum(p, 0, v)
        for (i in 0 until v) p[i] /= sum
        val k = minOf(topK, v)
        val idx = IntArray(k) { -1 }; val pk = DoubleArray(k) { -1.0 }
        for (i in 0 until v) {
            val q = p[i]
            if (q > pk[k - 1]) {
                var a = k - 1
                while (a > 0 && q > pk[a - 1]) { pk[a] = pk[a - 1]; idx[a] = idx[a - 1]; a-- }
                pk[a] = q; idx[a] = i
            }
        }
        val c = DoubleArray(k)
        var cum = 0.0; var acc = 0.0
        for (i in 0 until k) {
            val keep = i == 0 || cum < topP              // (cumsum - vals) < top_p
            cum += pk[i]; if (keep) acc += pk[i]; c[i] = acc
        }
        var u = rng.uniform() * c[k - 1]
        var hit = k - 1
        for (i in 0 until k) if (c[i] > u) { hit = i; break }
        var j = idx[hit]
        val need = maxOf(1, (win * tauR).toInt())
        if (recent.isNotEmpty()) {
            var n = 0
            for (t in maxOf(0, recent.size - win) until recent.size) if (recent[t] == j) n++
            if (n >= need) {
                val cp = DoubleArray(v); var a = 0.0
                for (i in 0 until v) { a += p[i]; cp[i] = a }
                u = rng.uniform() * cp[v - 1]
                j = v - 1
                for (i in 0 until v) if (cp[i] > u) { j = i; break }
            }
        }
        return j
    }

    // ---- MeanFlow + decoder --------------------------------------------------------------------------------------
    fun meanflow(v: Voice, ids: IntArray, rng: Rng): LongArray {
        val nd = basis.size; val nl = vals.size
        val p = v.cR.size; val n = 2 * ids.size; val tot = p + n
        val content = LongArray(tot); val seg = LongArray(tot); val xv = FloatArray(tot * nd)
        for (i in 0 until p) {
            content[i] = v.cR[i]
            for (d in 0 until nd) xv[i * nd + d] = vals[((v.acR[i] / basis[d]) % nl).toInt()]
        }
        val noise = rng.normal(n * nd)
        val s2n = cfg["s2_noise"].num().toFloat()
        for (i in 0 until n) {
            content[p + i] = ids[i / 2].toLong(); seg[p + i] = 1
            for (d in 0 until nd) { noise[i * nd + d] = noise[i * nd + d] * s2n; xv[(p + i) * nd + d] = noise[i * nd + d] }
        }
        val out = LongArray(n)
        OnnxTensor.createTensor(env, LongBuffer.wrap(content), longArrayOf(1, tot.toLong())).use { tc ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(xv), longArrayOf(1, tot.toLong(), nd.toLong())).use { tx ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(seg), longArrayOf(1, tot.toLong())).use { ts ->
                    mf.run(mapOf("content" to tc, "xv" to tx, "seg" to ts), setOf("u")).use { r ->
                        val u = FloatArray(tot * nd); (r[0] as OnnxTensor).floatBuffer.get(u)
                        for (i in 0 until n) {
                            var id = 0L
                            for (d in 0 until nd) {
                                val x1 = noise[i * nd + d] + u[(p + i) * nd + d]
                                var best = 0; var bd = Float.MAX_VALUE
                                for (l in 0 until nl) { val dd = Math.abs(x1 - vals[l]); if (dd < bd) { bd = dd; best = l } }
                                id += best * basis[d]
                            }
                            out[i] = id
                        }
                    }
                }
            }
        }
        return out
    }

    fun decode(ids: IntArray, aHat: LongArray, n24: Int): FloatArray {
        val nFft = cfg["n_fft"].int(); val hop = cfg["hop"].int()
        val s = LongArray(ids.size) { ids[it].toLong() }
        val y = OnnxTensor.createTensor(env, LongBuffer.wrap(s), longArrayOf(1, s.size.toLong())).use { a ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(aHat), longArrayOf(1, aHat.size.toLong())).use { b ->
                dec.run(mapOf("idx_s" to a, "idx_a" to b), setOf("logits")).use { r ->
                    val t = r[0] as OnnxTensor
                    val shp = t.info.shape
                    val rows = FloatArray((shp[1] * shp[2]).toInt()); t.floatBuffer.get(rows)
                    Istft.run(rows, shp[1].toInt(), nFft, hop)
                }
            }
        }
        return if (y.size >= n24) y.copyOf(n24) else y.copyOf(n24)
    }

    // ---- trim + watermark ----------------------------------------------------------------------------------------
    /** torchaudio sinc resample 24 -> 16 kHz with the fixed kernel (consts.npz). */
    fun resample2416(y: FloatArray): FloatArray {
        val o = 3; val n = 2; val w = rsWidth; val K = rsK
        val yp = FloatArray(y.size + 2 * w + o)
        System.arraycopy(y, 0, yp, w, y.size)
        val nfr = (yp.size - K) / o + 1
        val outLen = Math.ceil(y.size * n / o.toDouble()).toInt()
        val out = FloatArray(nfr * n)
        for (f in 0 until nfr) for (c in 0 until n) {
            var acc = 0f
            val base = f * o
            for (k in 0 until K) acc += yp[base + k] * rsKernel[c * K + k]
            out[f * n + c] = acc
        }
        return out.copyOf(minOf(outLen, out.size))
    }

    /** Silero VAD -> (duration, lead, tail) in seconds (= get_speech_timestamps: 300 ms silence, 30 ms pad, 0.1 s rounding). */
    fun speechBounds(y24: FloatArray): Triple<Double, Double, Double> { val b = bounds(y24); return Triple(b[0], b[1], b[2]) }

    /** speechBounds + the longest gap between speech spans: [dur, lead, tail, gap]. */
    fun bounds(y24: FloatArray): DoubleArray {
        val x = resample2416(y24)
        val sr16 = 16000; val ws = 512; val ctxN = 64
        var st = FloatArray(2 * 128)
        var ctx = FloatArray(ctxN)
        val probs = ArrayList<Float>()
        val srT = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(16000L)), LongArray(0))
        srT.use {
            var i = 0
            while (i < x.size) {
                val inp = FloatArray(ctxN + ws)
                System.arraycopy(ctx, 0, inp, 0, ctxN)
                System.arraycopy(x, i, inp, ctxN, minOf(ws, x.size - i))
                OnnxTensor.createTensor(env, FloatBuffer.wrap(inp), longArrayOf(1, (ctxN + ws).toLong())).use { ti ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(st), longArrayOf(2, 1, 128)).use { ts ->
                        vad!!.run(mapOf("input" to ti, "state" to ts, "sr" to srT)).use { r ->
                            val o = FloatArray(1); (r[0] as OnnxTensor).floatBuffer.get(o); probs.add(o[0])
                            val ns = FloatArray(256); (r[1] as OnnxTensor).floatBuffer.get(ns); st = ns
                        }
                    }
                }
                ctx = inp.copyOfRange(ws, ws + ctxN)
                i += ws
            }
        }
        val thr = 0.5f; val neg = 0.35f
        val minSp = sr16 * 0.25; val pad = sr16 * 0.03; val minSil = sr16 * 0.3
        val n = x.size
        var trig = false; var tempEnd = 0; var cur: IntArray? = null
        val sp = ArrayList<IntArray>()
        for ((i, pr) in probs.withIndex()) {
            val s = ws * i
            if (pr >= thr && tempEnd != 0) tempEnd = 0
            if (pr >= thr && !trig) { trig = true; cur = intArrayOf(s, -1); continue }
            if (pr < neg && trig) {
                if (tempEnd == 0) tempEnd = s
                if (s - tempEnd < minSil) continue
                cur!![1] = tempEnd
                if (cur[1] - cur[0] > minSp) sp.add(cur)
                cur = null; tempEnd = 0; trig = false
            }
        }
        if (cur != null && n - cur[0] > minSp) { cur[1] = n; sp.add(cur) }
        for (i in sp.indices) {
            val a = sp[i]
            if (i == 0) a[0] = maxOf(0.0, a[0] - pad).toInt()
            if (i != sp.size - 1) {
                val gap = sp[i + 1][0] - a[1]
                if (gap < 2 * pad) { a[1] += gap / 2; sp[i + 1][0] = maxOf(0, sp[i + 1][0] - gap / 2) }
                else { a[1] = minOf(n.toDouble(), a[1] + pad).toInt(); sp[i + 1][0] = maxOf(0.0, sp[i + 1][0] - pad).toInt() }
            } else a[1] = minOf(n.toDouble(), a[1] + pad).toInt()
        }
        val dur = n / sr16.toDouble()
        if (sp.isEmpty()) return doubleArrayOf(r2(dur), r2(dur), 0.0, 0.0)
        val start = maxOf(pyRound(sp[0][0] / sr16.toDouble(), 1), 0.0)
        val end = minOf(pyRound(sp.last()[1] / sr16.toDouble(), 1), dur)
        var gap = 0.0
        for (i in 0 until sp.size - 1) gap = maxOf(gap, pyRound(sp[i + 1][0] / sr16.toDouble(), 1) - pyRound(sp[i][1] / sr16.toDouble(), 1))
        return doubleArrayOf(r2(dur), r2(start), r2(dur - end), r2(gap))
    }

    fun trimEdges(y: FloatArray): FloatArray {
        val pad = cfg["trim_pad"].num()
        val (dur, lead, tail) = speechBounds(y)
        val a = maxOf(0, ((lead - pad) * sr).toInt()); val b = minOf(y.size, ((dur - tail + pad) * sr).toInt())
        return if (b > a) y.copyOfRange(a, b) else FloatArray(0)
    }

    /** AudioSeal mark: fixed 2 s graph, the clip in 2 s chunks (last zero-padded), cut back to length. */
    fun mark(y: FloatArray): FloatArray {
        if (y.isEmpty() || wm == null) return y
        val ch = cfg["wm_chunk"].int()
        val nCh = (y.size + ch - 1) / ch
        val out = FloatArray(nCh * ch)
        val seg = FloatArray(ch)
        for (c in 0 until nCh) {
            java.util.Arrays.fill(seg, 0f)
            System.arraycopy(y, c * ch, seg, 0, minOf(ch, y.size - c * ch))
            OnnxTensor.createTensor(env, FloatBuffer.wrap(seg), longArrayOf(1, ch.toLong())).use { t ->
                wm.run(mapOf("y" to t)).use { r -> (r[0] as OnnxTensor).floatBuffer.get(out, c * ch, ch) }
            }
        }
        return out.copyOf(y.size)
    }

    companion object {
        fun concat(parts: List<FloatArray>): FloatArray {
            val out = FloatArray(parts.sumOf { it.size }); var o = 0
            for (p in parts) { System.arraycopy(p, 0, out, o, p.size); o += p.size }
            return out
        }
        /** numpy pairwise summation (float64, contiguous): blocks of 8, split above 128. */
        fun pairwiseSum(a: DoubleArray, off: Int, n: Int): Double {
            if (n < 8) { var s = -0.0; for (i in 0 until n) s += a[off + i]; return s }
            if (n <= 128) {
                val r = DoubleArray(8) { a[off + it] }
                var i = 8
                while (i < n - (n % 8)) { for (j in 0 until 8) r[j] += a[off + i + j]; i += 8 }
                var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                while (i < n) { res += a[off + i]; i++ }
                return res
            }
            var n2 = n / 2; n2 -= n2 % 8
            return pairwiseSum(a, off, n2) + pairwiseSum(a, off + n2, n - n2)
        }
        /** python round(x, nd): round half to even on the decimal value. */
        fun pyRound(x: Double, nd: Int): Double = java.math.BigDecimal(x).setScale(nd, java.math.RoundingMode.HALF_EVEN).toDouble()
        fun r2(x: Double): Double = pyRound(x, 2)
    }
}

/** Script runs, language guess, unsupported scripts (= dvitts_onnx.runs / guess_lang / unsupported_scripts). */
object Script {
    private val BLOCKS = listOf(Triple("deva", 0x0900, 0x097F), Triple("beng", 0x0980, 0x09FF), Triple("guru", 0x0A00, 0x0A7F),
                                Triple("gujr", 0x0A80, 0x0AFF), Triple("orya", 0x0B00, 0x0B7F))
    private val LANG_OF = mapOf("latin" to "en", "deva" to "hi", "beng" to "bn", "guru" to "pa", "gujr" to "gu", "orya" to "or")
    const val SPOKEN = "English, Hindi, Marathi, Bengali, Punjabi, Gujarati, Odia"

    fun runs(text: String): List<Pair<String, String>> {
        val out = ArrayList<Pair<String, StringBuilder>>()
        text.codePoints().forEach { cp ->
            val s = BLOCKS.firstOrNull { cp in it.second..it.third }?.first ?: if (Character.isLetter(cp)) "latin" else null
            if (out.isNotEmpty() && (s == null || s == out.last().first)) out.last().second.appendCodePoint(cp)
            else if (s != null) out.add(s to StringBuilder().appendCodePoint(cp))
            else if (out.isEmpty()) out.add("latin" to StringBuilder().appendCodePoint(cp))
        }
        return out.map { it.first to it.second.toString() }
    }

    fun guessLang(text: String): String {
        val n = LinkedHashMap<String, Int>()
        for ((s, r) in runs(text)) n[s] = (n[s] ?: 0) + r.codePoints().toArray().count { Character.isLetter(it) || it in 0x0900..0x0B7F }
        return if (n.isEmpty()) "en" else LANG_OF.getValue(n.maxByOrNull { it.value }!!.key)
    }

    fun unsupported(text: String): List<String> {
        val bad = sortedSetOf<String>()
        text.codePoints().forEach { o ->
            if (!Character.isLetter(o) || o <= 0x024F || BLOCKS.any { o in it.second..it.third } || o in 0x1E00..0x1EFF) return@forEach
            bad.add((Character.getName(o) ?: "U+%04X".format(o)).split(" ")[0])
        }
        return bad.toList()
    }
}
