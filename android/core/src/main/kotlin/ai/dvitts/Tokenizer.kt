package ai.dvitts

import java.io.File
import java.text.Normalizer

/**
 * Qwen2 byte-level BPE from a HF tokenizer.json (= tokenizers' encode(text, add_special_tokens=False)):
 * added tokens matched first (leftmost, longest), the rest NFC-normalized, split by the Qwen2 pre-tokenizer regex,
 * mapped to GPT-2 byte characters, then BPE merges by rank.
 */
class Tokenizer(file: File) {
    private val vocab: HashMap<String, Int>
    private val ranks: HashMap<Long, Int>               // (left id, right id) -> merge rank
    private val mergedId: HashMap<Long, Int>
    private val added: List<Pair<String, Int>>          // longest first
    private val addedFirst: Set<Char>
    private val byteChar = CharArray(256)
    private val cache = HashMap<String, IntArray>()

    // Qwen2 split pattern; \s spelled as Unicode White_Space (oniguruma), \p{L} \p{N} as in the file
    private val ws = "[\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000]"
    private val pat = Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^${ws.substring(1, ws.length - 1)}\\p{L}\\p{N}]+[\\r\\n]*" +
                            "|$ws*[\\r\\n]+|$ws+(?![^${ws.substring(1, ws.length - 1)}])|$ws+")

    init {
        val j = Json.parse(file.readText()).obj()
        val model = j["model"].obj()
        check(model["type"] == "BPE") { "tokenizer: model ${model["type"]}" }
        val v = model["vocab"].obj()
        vocab = HashMap(v.size * 2)
        for ((k, id) in v) vocab[k] = id.int()
        val ad = j["added_tokens"].list().map { it.obj() }
        for (a in ad) vocab[a["content"] as String] = a["id"].int()
        added = ad.map { (it["content"] as String) to it["id"].int() }.sortedByDescending { it.first.length }
        addedFirst = added.map { it.first[0] }.toSet()
        val merges = model["merges"].list()
        ranks = HashMap(merges.size * 2); mergedId = HashMap(merges.size * 2)
        for ((r, m) in merges.withIndex()) {
            val (a, b) = if (m is String) m.split(" ", limit = 2) else m.list().map { it as String }
            val ia = vocab[a] ?: continue; val ib = vocab[b] ?: continue; val im = vocab[a + b] ?: continue
            val key = (ia.toLong() shl 32) or ib.toLong()
            if (!ranks.containsKey(key)) { ranks[key] = r; mergedId[key] = im }
        }
        // GPT-2 bytes_to_unicode
        val bs = ArrayList<Int>()
        bs.addAll('!'.code..'~'.code); bs.addAll('¡'.code..'¬'.code); bs.addAll('®'.code..'ÿ'.code)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0 until 256) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        for (i in bs.indices) byteChar[bs[i]] = cs[i].toChar()
    }

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        var i = 0; var seg = 0
        while (i < text.length) {
            if (text[i] in addedFirst) {
                val hit = added.firstOrNull { text.startsWith(it.first, i) }
                if (hit != null) {
                    if (seg < i) encodePlain(text.substring(seg, i), out)
                    out.add(hit.second)
                    i += hit.first.length; seg = i
                    continue
                }
            }
            i++
        }
        if (seg < text.length) encodePlain(text.substring(seg), out)
        return out.toIntArray()
    }

    private fun encodePlain(s0: String, out: MutableList<Int>) {
        val s = Normalizer.normalize(s0, Normalizer.Form.NFC)
        var last = 0
        for (m in pat.findAll(s)) {
            if (m.range.first > last) piece(s.substring(last, m.range.first), out)
            piece(m.value, out)
            last = m.range.last + 1
        }
        if (last < s.length) piece(s.substring(last), out)
    }

    private fun piece(p: String, out: MutableList<Int>) {
        cache[p]?.let { c -> c.forEach { out.add(it) }; return }
        val bytes = p.toByteArray(Charsets.UTF_8)
        var ids = IntArray(bytes.size) { vocab.getValue(byteChar[bytes[it].toInt() and 0xff].toString()) }.toMutableList()
        while (ids.size > 1) {
            var best = Int.MAX_VALUE; var at = -1
            for (k in 0 until ids.size - 1) {
                val r = ranks[(ids[k].toLong() shl 32) or ids[k + 1].toLong()] ?: continue
                if (r < best) { best = r; at = k }
            }
            if (at < 0) break
            val key = (ids[at].toLong() shl 32) or ids[at + 1].toLong()
            val nid = mergedId.getValue(key)
            val nx = ArrayList<Int>(ids.size)
            var k = 0
            while (k < ids.size) {
                if (k < ids.size - 1 && ((ids[k].toLong() shl 32) or ids[k + 1].toLong()) == key) { nx.add(nid); k += 2 }
                else { nx.add(ids[k]); k++ }
            }
            ids = nx
        }
        val arr = ids.toIntArray()
        if (cache.size < 50_000) cache[p] = arr
        arr.forEach { out.add(it) }
    }
}
