package ai.dvitts

/**
 * Cardinal number words, = frontend.num_words (python):
 *  en  English rules written here (En; same output format as num2words, which the python front end imports), "," removed and "-" -> " "
 *  hi mr gu bn pa or  indic-numtowords 1.1.0 as tables (frontend_numwords.json, built from the library and checked
 *      against it): n < 1000 standalone[n]; n < 10^7 [direct[lakh] lakh] [direct[th] thousand] inner[r];
 *      n >= 10^7 "<words(n / 10^7)> crore [words(n % 10^7)]" (frontend.num_words' own recursion).
 * Values above Long range (19+ digits) are read digit by digit (python would spell the full number).
 */
class NumWords(tables: Map<String, Any?>) {
    private class Lang(t: Map<String, Any?>) {
        val direct = t["direct"].list().map { it as String }
        val higher = t["higher"].obj().mapValues { it.value as String }
        val standalone = t["standalone"].list().map { it as String }
        val inner = t["inner"].list().map { it as String }
        val exceptions = t["exceptions"].obj().mapValues { it.value as String }
        val overrides = (t["overrides"] ?: emptyMap<String, Any?>()).obj().mapValues { it.value as String }
    }

    private val langs = tables.filterKeys { it != "en" }.mapValues { Lang(it.value.obj()) }
    val decadeYears: Map<String, String> = tables["en"].obj()["decade_years"].obj().mapValues { it.value as String }

    fun words(n: Long, lang: String): String {
        require(n >= 0) { "num_words: negative $n" }
        if (lang == "en") return En.cardinal(n).replace(",", "").replace("-", " ")
        val crore = Frontend.CRORE.getValue(lang)
        if (n >= 10_000_000L) {
            val hi = n / 10_000_000L; val lo = n % 10_000_000L
            return "${words(hi, lang)} $crore" + (if (lo != 0L) " ${words(lo, lang)}" else "")
        }
        val t = langs.getValue(lang)
        val k = n.toString()
        t.overrides[k]?.let { return it }
        if (n < 1000) return t.standalone[n.toInt()]
        t.exceptions[k]?.let { return it }
        val lakh = (n / 100_000).toInt(); val rest = (n % 100_000).toInt()
        val th = rest / 1000; val r = rest % 1000
        val parts = ArrayList<String>()
        if (lakh != 0) parts.add("${t.direct[lakh]} ${t.higher.getValue("5")}")
        if (th != 0) parts.add("${t.direct[th]} ${t.higher.getValue("3")}")
        if (r != 0) parts.add(t.inner[r])
        return parts.joinToString(" ")
    }

    /** Digit string of any length: Long when it fits, else digit by digit. */
    fun words(digits: String, lang: String): String {
        val v = digits.toLongOrNull()
        return if (v != null) words(v, lang) else digits.filter { it.isDigit() }.map { words(Character.digit(it, 10).toLong(), lang) }.joinToString(" ")
    }

    /**
     * English cardinal numbers, written independently (kept independent so the library stays Apache 2.0). Plain rules:
     * under 20 one word; under 100 "tens-unit"; under 1000 "<d> hundred[ and <rest>]"; above, the largest scale
     * (thousand, million, ...) as "<head> <scale>" then the rest, joined by " and " when it is under 100 and by ", "
     * otherwise. That is the output format frontend.num_words was built on, so the golden vectors still apply.
     */
    object En {
        private val LOW = arrayOf("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
        private val TENS = arrayOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
        private val SCALES = listOf(1_000_000_000_000_000_000L to "quintillion", 1_000_000_000_000_000L to "quadrillion",
            1_000_000_000_000L to "trillion", 1_000_000_000L to "billion", 1_000_000L to "million", 1000L to "thousand")

        fun cardinal(n: Long): String {
            require(n >= 0)
            if (n < 20) return LOW[n.toInt()]
            if (n < 100) return TENS[(n / 10).toInt()] + (if (n % 10 != 0L) "-" + LOW[(n % 10).toInt()] else "")
            if (n < 1000) return LOW[(n / 100).toInt()] + " hundred" + (if (n % 100 != 0L) " and " + cardinal(n % 100) else "")
            val (scale, word) = SCALES.first { it.first <= n }
            val head = cardinal(n / scale) + " " + word
            val rest = n % scale
            return when {
                rest == 0L -> head
                rest < 100 -> "$head and ${cardinal(rest)}"
                else -> "$head, ${cardinal(rest)}"
            }
        }
    }
}
