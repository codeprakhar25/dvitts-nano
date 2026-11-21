package ai.dvitts

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/** Minimal JSON reader: objects -> Map<String, Any?>, arrays -> List<Any?>, numbers -> Long or Double. */
object Json {
    fun parse(s: String): Any? = P(s).run { val v = value(); ws(); v }

    private class P(val s: String) {
        var i = 0
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            ws()
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> { i += 4; true }
                'f' -> { i += 5; false }
                'n' -> { i += 4; null }
                else -> if (c == '-' || c.isDigit()) num() else error("json: bad char '$c' at $i")
            }
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++; ws()
            if (s[i] == '}') { i++; return m }
            while (true) {
                ws(); val k = str(); ws()
                check(s[i] == ':') { "json: ':' expected at $i" }; i++
                m[k] = value(); ws()
                if (s[i] == ',') { i++; continue }
                check(s[i] == '}') { "json: '}' expected at $i" }; i++
                return m
            }
        }
        fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            i++; ws()
            if (s[i] == ']') { i++; return l }
            while (true) {
                l.add(value()); ws()
                if (s[i] == ',') { i++; continue }
                check(s[i] == ']') { "json: ']' expected at $i" }; i++
                return l
            }
        }
        fun str(): String {
            check(s[i] == '"'); i++
            val b = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return b.toString()
                    '\\' -> {
                        when (val e = s[i++]) {
                            'n' -> b.append('\n'); 't' -> b.append('\t'); 'r' -> b.append('\r')
                            'b' -> b.append('\b'); 'f' -> b.append('\u000c')
                            'u' -> { b.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                            else -> b.append(e)
                        }
                    }
                    else -> b.append(c)
                }
            }
        }
        fun num(): Any {
            val st = i
            if (s[i] == '-') i++
            while (i < s.length && (s[i].isDigit() || s[i] in ".eE+-")) i++
            val t = s.substring(st, i)
            return if (t.any { it in ".eE" }) t.toDouble() else t.toLongOrNull() ?: t.toDouble()
        }
    }
}

@Suppress("UNCHECKED_CAST")
fun Any?.obj(): Map<String, Any?> = this as Map<String, Any?>
@Suppress("UNCHECKED_CAST")
fun Any?.list(): List<Any?> = this as List<Any?>
fun Any?.num(): Double = (this as Number).toDouble()
fun Any?.int(): Int = (this as Number).toInt()

/** One array from an .npz (numpy savez): little-endian int64 / float32 / float64, C order. */
class NpArray(val shape: IntArray, val dtype: String, val buf: ByteBuffer) {
    val size: Int get() = shape.fold(1) { a, b -> a * b }
    fun longs(): LongArray = when (dtype) {
        "<i8" -> LongArray(size).also { buf.asLongBuffer().get(it) }
        "<i4" -> IntArray(size).also { buf.asIntBuffer().get(it) }.map { it.toLong() }.toLongArray()
        else -> error("npy: $dtype is not an integer type")
    }
    fun floats(): FloatArray = when (dtype) {
        "<f4" -> FloatArray(size).also { buf.asFloatBuffer().get(it) }
        "<f8" -> DoubleArray(size).also { buf.asDoubleBuffer().get(it) }.map { it.toFloat() }.toFloatArray()
        else -> error("npy: $dtype is not a float type")
    }
    fun doubles(): DoubleArray = when (dtype) {
        "<f8" -> DoubleArray(size).also { buf.asDoubleBuffer().get(it) }
        "<f4" -> FloatArray(size).also { buf.asFloatBuffer().get(it) }.map { it.toDouble() }.toDoubleArray()
        else -> error("npy: $dtype is not a float type")
    }
}

object Npz {
    fun read(f: File): Map<String, NpArray> = ZipFile(f).use { z ->
        z.entries().toList().filter { it.name.endsWith(".npy") }.associate { e ->
            e.name.removeSuffix(".npy") to npy(z.getInputStream(e))
        }
    }

    private fun npy(ins: InputStream): NpArray {
        val all = ins.readBytes()
        check(all[0] == 0x93.toByte() && String(all, 1, 5, Charsets.US_ASCII) == "NUMPY") { "not an .npy" }
        val major = all[6].toInt()
        val (hlen, hoff) = if (major == 1) (all[8].toInt() and 0xff or ((all[9].toInt() and 0xff) shl 8)) to 10
                           else (ByteBuffer.wrap(all, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int) to 12
        val header = String(all, hoff, hlen, Charsets.US_ASCII)
        val descr = Regex("'descr':\\s*'([^']+)'").find(header)!!.groupValues[1]
        check(!header.contains("'fortran_order': True")) { "npy: fortran order not supported" }
        val shp = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)!!.groupValues[1]
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray()
        val start = hoff + hlen
        val buf = ByteBuffer.wrap(all, start, all.size - start).slice().order(ByteOrder.LITTLE_ENDIAN)
        return NpArray(shp, descr, buf)
    }
}

/** Code-point helpers: Python counts and indexes strings by code point; Kotlin by UTF-16 unit. */
fun String.cpLen(): Int = codePointCount(0, length)
