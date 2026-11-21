package ai.dvitts

import java.text.Normalizer
import java.time.DateTimeException
import java.time.LocalDate

/**
 * Text front end, = frontend.py on its portable path (English without NeMo, as dvitts_onnx.normalize).
 * normalize(text, lang): written form -> spoken form; split(text, lang): chunks for long text.
 *
 * Regex notes (python re -> java.util.regex): python \w is [\p{L}\p{N}_] (no combining marks), \d is \p{Nd},
 * \s is str.isspace; java's own classes differ, so they are spelled out below, and python \b becomes a lookaround on W.
 */
class Frontend(private val nw: NumWords) {
    companion object {
        val LANGS = listOf("en", "hi", "mr", "gu", "bn", "pa", "or")
        val MAX_CHARS = mapOf("en" to 500); const val MAX_CHARS_IN = 400
        val MIN_CHARS = mapOf("en" to 300); const val MIN_CHARS_IN = 250
        private val DIGIT_ZERO = intArrayOf(0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66)

        const val W = "[\\p{L}\\p{N}_]"
        const val NW = "[^\\p{L}\\p{N}_]"
        const val D = "\\p{Nd}"
        const val SP = "\\t\\n\\u000B\\f\\r\\u001C-\\u001F \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"
        const val S = "[$SP]"
        const val NS = "[^$SP]"
        const val BB = "(?<!$W)"          // python \b before a word char
        const val BA = "(?!$W)"           // python \b after a word char

        val WORDS: Map<String, Map<String, String>> = mapOf(
            "hi" to mapOf("rupee" to "रुपये", "dollar" to "डॉलर", "percent" to "प्रतिशत", "point" to "दशमलव", "dot" to "डॉट", "at" to "एट",
                "slash" to "स्लैश", "plus" to "प्लस", "and" to "और", "baje" to "बजे", "bajkar" to "बजकर", "minute" to "मिनट", "hundred" to "सौ"),
            "mr" to mapOf("rupee" to "रुपये", "dollar" to "डॉलर", "percent" to "टक्के", "point" to "पूर्णांक", "dot" to "डॉट", "at" to "एट",
                "slash" to "स्लॅश", "plus" to "प्लस", "and" to "आणि", "minute" to "मिनिटे"),
            "gu" to mapOf("rupee" to "રૂપિયા", "dollar" to "ડોલર", "percent" to "ટકા", "point" to "દશાંશ", "dot" to "ડોટ", "at" to "એટ",
                "slash" to "સ્લેશ", "plus" to "પ્લસ", "and" to "અને", "minute" to "મિનિટ"),
            "bn" to mapOf("rupee" to "টাকা", "dollar" to "ডলার", "percent" to "শতাংশ", "point" to "দশমিক", "dot" to "ডট", "at" to "অ্যাট",
                "slash" to "স্ল্যাশ", "plus" to "প্লাস", "and" to "এবং", "minute" to "মিনিট"),
            "pa" to mapOf("rupee" to "ਰੁਪਏ", "dollar" to "ਡਾਲਰ", "percent" to "ਪ੍ਰਤੀਸ਼ਤ", "point" to "ਦਸ਼ਮਲਵ", "dot" to "ਡੌਟ", "at" to "ਐਟ",
                "slash" to "ਸਲੈਸ਼", "plus" to "ਪਲੱਸ", "and" to "ਅਤੇ", "minute" to "ਮਿੰਟ"),
            "or" to mapOf("rupee" to "ଟଙ୍କା", "dollar" to "ଡଲାର", "percent" to "ପ୍ରତିଶତ", "point" to "ଦଶମିକ", "dot" to "ଡଟ", "at" to "ଆଟ",
                "slash" to "ସ୍ଲାସ", "plus" to "ପ୍ଲସ", "and" to "ଓ", "minute" to "ମିନିଟ"),
        )
        val MONTHS: Map<String, List<String>> = mapOf(
            "hi" to "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(" "),
            "mr" to "जानेवारी फेब्रुवारी मार्च एप्रिल मे जून जुलै ऑगस्ट सप्टेंबर ऑक्टोबर नोव्हेंबर डिसेंबर".split(" "),
            "gu" to "જાન્યુઆરી ફેબ્રુઆરી માર્ચ એપ્રિલ મે જૂન જુલાઈ ઓગસ્ટ સપ્ટેમ્બર ઓક્ટોબર નવેમ્બર ડિસેમ્બર".split(" "),
            "bn" to "জানুয়ারি ফেব্রুয়ারি মার্চ এপ্রিল মে জুন জুলাই আগস্ট সেপ্টেম্বর অক্টোবর নভেম্বর ডিসেম্বর".split(" "),
            "pa" to "ਜਨਵਰੀ ਫ਼ਰਵਰੀ ਮਾਰਚ ਅਪ੍ਰੈਲ ਮਈ ਜੂਨ ਜੁਲਾਈ ਅਗਸਤ ਸਤੰਬਰ ਅਕਤੂਬਰ ਨਵੰਬਰ ਦਸੰਬਰ".split(" "),
            "or" to "ଜାନୁଆରୀ ଫେବୃଆରୀ ମାର୍ଚ୍ଚ ଅପ୍ରେଲ ମେ ଜୁନ ଜୁଲାଇ ଅଗଷ୍ଟ ସେପ୍ଟେମ୍ବର ଅକ୍ଟୋବର ନଭେମ୍ବର ଡିସେମ୍ବର".split(" "),
        )
        val ABBR: Map<String, List<Pair<String, String>>> = mapOf(
            "hi" to listOf("डॉ." to "डॉक्टर", "किमी" to "किलोमीटर", "कि.मी." to "किलोमीटर", "सेमी" to "सेंटीमीटर", "किग्रा" to "किलोग्राम",
                "मि.ली." to "मिलीलीटर", "रु." to "रुपये"),
            "mr" to listOf("डॉ." to "डॉक्टर", "किमी" to "किलोमीटर"))
        val CRORE = mapOf("hi" to "करोड़", "mr" to "कोटी", "gu" to "કરોડ", "bn" to "কোটি", "pa" to "ਕਰੋੜ", "or" to "କୋଟି")
        private val EN_WORDS = mapOf("dot" to "dot", "at" to "at", "slash" to "slash")
        val MONTHS_EN = "January February March April May June July August September October November December".split(" ")

        const val NUM = "$D{1,3}(?:,$D{2,3})+|$D+"
        private val DATE_WORDS = mapOf(
            "en" to "dated?|dates|dob|d\\.o\\.b\\.?|born|on|birthday|due|expir$W*|valid (?:till|until|upto)|from|till|until",
            "in" to "दिनांक|तारीख़?|तिथि|जन्मतिथि|जन्म तिथि|dob|date")

        private val R_URL = Regex("(https?://$NS+|www\\.$NS+)")
        private val R_EMAIL = Regex("[\\p{L}\\p{N}_.+-]+@[\\p{L}\\p{N}_-]+(?:\\.[\\p{L}\\p{N}_-]+)+")
        private val R_DMY = Regex("$BB($D{1,2})[/.-]($D{1,2})[/.-]($D{4})$BA")
        private val R_CLOCK = Regex("$BB($D{1,2}):($D{2})$BA(?:$S*बजे)?")
        private val R_PHONE = Regex("\\+?$D[${"\\p{Nd}"} -]{8,}$D")
        private const val MONEY = "(?:$NUM)(?:\\.$D+)?"
        private val R_RUPEE = Regex("(?:₹|${BB}Rs\\.?|${BB}INR)$S?($MONEY)")
        private val R_DOLLAR = Regex("\\$$S?($MONEY)")
        private val R_PCT = Regex("($MONEY)$S?%")
        private val R_DEC = Regex("($NUM)\\.($D+)")
        private val R_AND_D = Regex("($D)$S*&$S*($D)")
        private val R_AND = Regex("$S&$S")
        private val R_NUM = Regex(NUM)
        private val R_WWW = Regex("${BB}www$BA")
        private val R_DIGITS = Regex("$D+")
        private val R_SPACES = Regex("[ \\t]+")
        private val R_NL = Regex(" *\\n *")
        private val R_END = Regex("(?<=[.!?।;])[\"')\\]]*$S+|\\n+")
        private val R_COMMA = Regex("(?<=[,:،])$S+")
        private val R_WS = Regex("$S+")
        private val GAP = "(?:$NW{0,3}$S*(?:is|was|of|है|था|:)?$NW{0,3}$S*)"
        private val R_DATE8 = mapOf(
            "en" to Regex("(?iu)$BB(${DATE_WORDS["en"]})($GAP)($D{2})($D{2})($D{4})$BA"),
            "in" to Regex("(?iu)$BB(${DATE_WORDS["in"]})($GAP)($D{2})($D{2})($D{4})$BA"))
    }

    private fun w(lang: String, key: String): String = WORDS[lang]?.get(key) ?: EN_WORDS[key] ?: key
    private fun int(s: String): Long = s.toLong()
    private fun numWords(n: Long, lang: String) = nw.words(n, lang)
    private fun numWords(s: String, lang: String) = nw.words(s, lang)
    private fun digitsWords(s: String, lang: String): String =
        s.codePoints().toArray().filter { Character.isDigit(it) }.joinToString(" ") { numWords(Character.digit(it, 10).toLong(), lang) }

    private fun yearHi(n: Long): String {
        val hi = n / 100; val lo = n % 100
        return "${numWords(hi, "hi")} सौ" + (if (lo != 0L) " ${numWords(lo, "hi")}" else "")
    }

    /** URL or e-mail -> spoken pieces: letters kept, '.' '@' '/' as words. */
    private fun spellNet(s0: String, lang: String): String {
        val s = s0.replace(Regex("^https?://"), "").trimEnd('/')
        val out = ArrayList<String>()
        val part = StringBuilder()
        fun flush() {
            if (part.isNotEmpty()) {
                out.add(R_DIGITS.replace(part.toString()) { " " + digitsWords(it.value, lang) + " " }.pyStrip())
                part.setLength(0)
            }
        }
        for (c in s) {
            when (c) {
                '.' -> { flush(); out.add(w(lang, "dot")) }
                '/' -> { flush(); out.add(w(lang, "slash")) }
                '@' -> { flush(); out.add(w(lang, "at")) }
                else -> part.append(c)
            }
        }
        flush()
        return out.joinToString(" ")
    }

    private fun date8(t: String, lang: String): String = R_DATE8.getValue(if (lang == "en") "en" else "in").replace(t) { m ->
        val g = m.groupValues
        val ok = try { val y = g[5].toInt(); y in 1900..2099 && LocalDate.of(y, g[4].toInt(), g[3].toInt()) != null }
                 catch (e: DateTimeException) { false }
        if (ok) "${g[1]}${g[2]}${g[3]}/${g[4]}/${g[5]}" else m.value
    }

    private fun indic(text: String, lang: String): String {
        val sb = StringBuilder()
        text.codePoints().forEach { cp ->
            val z = DIGIT_ZERO.firstOrNull { cp in it..it + 9 }
            if (z != null) sb.append(('0' + (cp - z))) else sb.appendCodePoint(cp)
        }
        var t = sb.toString()
        for ((a, b) in ABBR[lang].orEmpty()) t = Regex("(?<!$W)${Regex.escape(a)}(?!$W)").replace(t) { b }
        t = R_URL.replace(t) { spellNet(it.value.trimEnd('.', ','), lang) }
        if (lang == "hi" || lang == "mr") t = R_WWW.replace(t) { "डब्ल्यू डब्ल्यू डब्ल्यू" }
        t = R_EMAIL.replace(t) { spellNet(it.value, lang) }
        t = date8(t, lang)
        t = R_DMY.replace(t) { m ->
            val d = int(m.groupValues[1]); val mo = int(m.groupValues[2]); val y = m.groupValues[3]
            if (d !in 1..31 || mo !in 1..12) m.value
            else {
                val yi = int(y)
                val yw = if (lang == "hi" && yi in 1100..1999) yearHi(yi) else numWords(yi, lang)
                "${numWords(d, lang)} ${MONTHS.getValue(lang)[(mo - 1).toInt()]} $yw"
            }
        }
        t = R_CLOCK.replace(t) { m ->
            val h = int(m.groupValues[1]); val mi = int(m.groupValues[2])
            when {
                h > 23 || mi > 59 -> m.value
                lang == "hi" -> if (mi == 0L) "${numWords(h, lang)} ${w(lang, "baje")}"
                                else "${numWords(h, lang)} ${w(lang, "bajkar")} ${numWords(mi, lang)} ${w(lang, "minute")}"
                else -> if (mi != 0L) "${numWords(h, lang)} ${numWords(mi, lang)}" else numWords(h, lang)
            }
        }
        t = R_PHONE.replace(t) { m ->
            val s = m.value
            if (s.count { it.isDigit() } >= 10) (if (s.startsWith("+")) w(lang, "plus") + " " else "") + digitsWords(s, lang) else s
        }
        t = R_RUPEE.replace(t) { "${it.groupValues[1]} ${w(lang, "rupee")}" }
        t = R_DOLLAR.replace(t) { "${it.groupValues[1]} ${w(lang, "dollar")}" }
        t = R_PCT.replace(t) { "${it.groupValues[1]} ${w(lang, "percent")}" }
        t = R_DEC.replace(t) { m ->
            "${numWords(m.groupValues[1].replace(",", ""), lang)} ${w(lang, "point")} ${digitsWords(m.groupValues[2], lang)}"
        }
        t = R_AND_D.replace(t) { "${it.groupValues[1]} ${w(lang, "and")} ${it.groupValues[2]}" }
        t = R_AND.replace(t) { " ${w(lang, "and")} " }
        t = R_NUM.replace(t) { m ->
            val s = m.value
            val digits = s.replace(",", "")
            val n = digits.toLongOrNull()
            if (lang == "hi" && !s.contains(",") && s.length == 4 && n != null && n in 1100..1999) yearHi(n) else numWords(digits, lang)
        }
        return t
    }

    private fun enPre(t0: String): String {
        var t = date8(t0, "en")
        t = R_DMY.replace(t) { m ->
            val d = int(m.groupValues[1]); val mo = int(m.groupValues[2])
            if (d in 1..31 && mo in 1..12) "${MONTHS_EN[(mo - 1).toInt()]} $d, ${m.groupValues[3]}" else m.value
        }
        t = R_URL.replace(t) { spellNet(it.value.trimEnd('.', ','), "en") }
        t = R_EMAIL.replace(t) { spellNet(it.value, "en") }
        t = R_WWW.replace(t) { "double-u double-u double-u" }
        t = R_PHONE.replace(t) { m ->
            val s = m.value
            if (s.count { it.isDigit() } >= 10) (if (s.startsWith("+")) "plus " else "") + s.filter { it.isDigit() }.toList().joinToString(" ") else s
        }
        t = Regex("(?:₹|${BB}Rs\\.?|${BB}INR)$S?((?:$NUM)(?:\\.$D+)?)").replace(t) { "${it.groupValues[1]} rupees" }
        t = Regex("#$S?($D+)").replace(t) { "number ${it.groupValues[1]}" }
        t = Regex("$S?°$S?C$BA").replace(t) { " degrees Celsius" }
        t = Regex("$S?°$S?F$BA").replace(t) { " degrees Fahrenheit" }
        t = Regex("${BB}km/h$BA|${BB}kmph$BA").replace(t) { "kilometers per hour" }
        t = Regex("${BB}mph$BA").replace(t) { "miles per hour" }
        t = Regex("$BB(1[0-9]{2}0|20[0-9]0)s$BA").replace(t) { m ->
            val wds = nw.decadeYears.getValue(m.groupValues[1]).replace("-", " ").split(" ").filter { it.isNotEmpty() }.toMutableList()
            val last = wds.last()
            wds[wds.size - 1] = if (last.endsWith("y")) last.dropLast(1) + "ies" else last + "s"
            wds.joinToString(" ")
        }
        t = Regex("$BB$D+(?:\\.$D+){2,}$BA").replace(t) { it.value.split(".").joinToString(" point ") }
        t = Regex("(?<![\\p{L}\\p{N}_-])-($D)").replace(t) { "minus ${it.groupValues[1]}" }
        t = Regex("$BB($D+)$S?-$S?($D+)($S+(?:days|weeks|months|years|hours|minutes|seconds|times|people|pages))").replace(t) {
            "${it.groupValues[1]} to ${it.groupValues[2]}${it.groupValues[3]}"
        }
        return t
    }

    private fun en(text: String): String {
        var t = enPre(text)
        t = R_URL.replace(t) { spellNet(it.value.trimEnd('.', ','), "en") }
        t = Regex("\\$$S?($D+)").replace(t) { "${it.groupValues[1]} dollars" }
        t = Regex("($D+)$S?%").replace(t) { "${it.groupValues[1]} percent" }
        return R_DIGITS.replace(t) { numWords(it.value, "en") }
    }

    fun normalize(text: String, lang: String): String {
        require(lang in LANGS) { "lang '$lang': one of $LANGS" }
        val n = Normalizer.normalize(text, Normalizer.Form.NFC)
        val sb = StringBuilder()
        n.codePoints().forEach { cp ->
            val ty = Character.getType(cp)
            if (cp == '°'.code || (ty != Character.OTHER_SYMBOL.toInt() && ty != Character.SURROGATE.toInt() && ty != Character.PRIVATE_USE.toInt()))
                sb.appendCodePoint(cp)
        }
        var t = sb.toString().replace("​", "").replace("॥", "।")
        t = t.replace('‘', '\'').replace('’', '\'').replace('“', '"').replace('”', '"')
        t = if (lang == "en") en(t) else indic(t, lang)
        t = R_SPACES.replace(t, " ")
        return R_NL.replace(t, "\n").pyStrip()
    }

    fun split(text: String, lang: String, maxChars: Int? = null, minChars: Int? = null): List<String> {
        val mx = maxChars ?: MAX_CHARS[lang] ?: MAX_CHARS_IN
        val mn = minChars ?: MIN_CHARS[lang] ?: MIN_CHARS_IN
        val pieces = ArrayList<String>()
        for (s0 in text.split(R_END)) {
            val s = s0.pyStrip()
            if (s.isEmpty()) continue
            if (s.cpLen() <= mx) { pieces.add(s); continue }
            var cur = ""
            for (c in s.split(R_COMMA)) {
                val words = if (c.cpLen() > mx) c.split(R_WS).filter { it.isNotEmpty() } else listOf(c)
                for (wd in words) {
                    if (cur.isNotEmpty() && cur.cpLen() + 1 + wd.cpLen() > mx) { pieces.add(cur); cur = wd }
                    else cur = "$cur $wd".pyStrip()
                }
            }
            if (cur.isNotEmpty()) pieces.add(cur)
        }
        val out = ArrayList<String>()
        for (p in pieces) {
            if (out.isNotEmpty() && (out.last().cpLen() < mn || p.cpLen() < mn / 3) && out.last().cpLen() + 1 + p.cpLen() <= mx)
                out[out.size - 1] = "${out.last()} $p"
            else out.add(p)
        }
        return out
    }
}

/** python str.strip(): str.isspace characters at both ends. */
fun String.pyStrip(): String {
    val ws = { c: Char -> c in "\t\n\u000B\u000c\r\u001C\u001D\u001E\u001F \u0085      　" || c in ' '..' ' }
    var a = 0; var b = length
    while (a < b && ws(this[a])) a++
    while (b > a && ws(this[b - 1])) b--
    return substring(a, b)
}
