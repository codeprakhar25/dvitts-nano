package ai.dvitts.android

import ai.dvitts.DviTTS
import ai.dvitts.Json
import ai.dvitts.obj
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Model folder on the device: a .zip downloaded or imported once into app storage, every file md5-checked against
 *  manifest.json. */
object ModelStore {
    const val DIR = "dvitts-model"

    fun dir(ctx: Context): File = File(ctx.filesDir, DIR)

    fun ready(ctx: Context): Boolean = File(dir(ctx), "config.json").exists() && File(dir(ctx), "manifest.json").exists()

    /** Model name from config.json ("DviTTS ONNX v1"), or null when no model is installed. */
    fun name(ctx: Context): String? = try { Json.parse(File(dir(ctx), "config.json").readText()).obj()["name"] as? String } catch (e: Exception) { null }

    /** Download + unpack; progress(percent) is called every 5 %. Throws on a bad link or a file that fails its md5. */
    fun download(ctx: Context, link: String, progress: (Int) -> Unit = {}): File {
        val c = URL(link).openConnection() as HttpURLConnection
        c.connectTimeout = 30_000; c.readTimeout = 60_000
        return unpack(ctx, c.inputStream, c.contentLengthLong, progress)
    }

    /** Unpack a model .zip from any stream (a file the user picked: Drive, WhatsApp, USB copy). total = its size or -1. */
    fun unpack(ctx: Context, input: InputStream, total: Long, progress: (Int) -> Unit = {}): File {
        val out = dir(ctx).apply { deleteRecursively(); mkdirs() }
        var got = 0L; var last = -1
        ZipInputStream(input.buffered(1 shl 16)).use { zin ->
            while (true) {
                val e = zin.nextEntry ?: break
                val name = e.name.substringAfter('/', e.name)
                if (e.isDirectory || name.isEmpty() || name.contains("..")) continue
                val f = File(out, name).also { it.parentFile?.mkdirs() }
                f.outputStream().buffered(1 shl 16).use { o ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val n = zin.read(buf); if (n < 0) break
                        o.write(buf, 0, n); got += n
                        val pct = if (total > 0) (100 * got / total).toInt() else -1
                        if (pct >= 0 && pct / 5 != last / 5) { last = pct; progress(pct) }
                    }
                }
            }
        }
        verify(out)
        return out
    }

    /** -> files that fail; empty = all good. */
    fun verify(d: File): List<String> {
        val man = Json.parse(File(d, "manifest.json").readText()).obj()
        // laptop_only files (llama.cpp stage 1, fp32 watermark) are not used on a phone: a phone zip may leave
        // them out, but a copy that is present must still match.
        val bad = man.filter { (name, v) ->
            val o = v.obj(); val f = File(d, name)
            !(o["laptop_only"] == true && !f.exists()) && md5(f) != o["md5"]
        }.keys.toList()
        check(bad.isEmpty()) { "model files failed their md5 check: $bad" }
        return bad
    }

    private fun md5(f: File): String {
        if (!f.exists()) return "missing"
        val md = MessageDigest.getInstance("MD5")
        f.inputStream().buffered(1 shl 20).use { s -> val b = ByteArray(1 shl 20); while (true) { val n = s.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun open(ctx: Context, threads: Int = 4, memMode: String = "all"): DviTTS = DviTTS(dir(ctx), threads = threads, memMode = memMode)
}

/** Plays 24 kHz mono float audio. */
object AudioOut {
    private var track: AudioTrack? = null

    fun play(y: FloatArray, sr: Int = 24000) {
        stop()
        val pcm = ShortArray(y.size) { (y[it].coerceIn(-1f, 1f) * 32767).toInt().toShort() }
        val tr = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT).setSampleRate(sr)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(pcm.size * 2, 4096)).setTransferMode(AudioTrack.MODE_STATIC).build()
        tr.write(pcm, 0, pcm.size)
        tr.play()
        track = tr
    }

    fun stop() { track?.run { try { stop() } catch (_: IllegalStateException) {}; release() }; track = null }

    /** Frames played so far by the current clip (0 when nothing is playing). */
    fun position(): Int = try { track?.playbackHeadPosition ?: 0 } catch (_: IllegalStateException) { 0 }

    /** 16-bit mono WAV bytes (to share or save a clip). */
    fun wav(y: FloatArray, sr: Int = 24000): ByteArray {
        val n = y.size * 2
        val b = java.nio.ByteBuffer.allocate(44 + n).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + n).put("WAVE".toByteArray()).put("fmt ".toByteArray()).putInt(16)
            .putShort(1).putShort(1).putInt(sr).putInt(sr * 2).putShort(2).putShort(16).put("data".toByteArray()).putInt(n)
        for (v in y) b.putShort((v.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        return b.array()
    }
}
