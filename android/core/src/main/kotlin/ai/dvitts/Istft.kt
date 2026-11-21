package ai.dvitts

/**
 * Vocos ISTFT head, 'same' padding (= dvitts_onnx.istft / tts_stream.StreamISTFT): rows [T, 2 * (nFft/2 + 1)] =
 * log-magnitude | phase -> magnitude clipped at 1e2 -> irfft (numpy scaling 1/n) x periodic Hann -> overlap-add,
 * divided by the summed squared window, (nFft - hop) / 2 cut from both ends. All in float64, float32 out.
 */
object Istft {
    private val ffts = HashMap<Int, Fft>()

    fun run(rows: FloatArray, t: Int, nFft: Int, hop: Int): FloatArray {
        val nb = nFft / 2 + 1
        require(rows.size == t * 2 * nb) { "istft: ${rows.size} values for $t frames x ${2 * nb}" }
        val fft = synchronized(ffts) { ffts.getOrPut(nFft) { Fft(nFft) } }
        val win = DoubleArray(nFft) { 0.5 - 0.5 * Math.cos(2 * Math.PI * it / nFft) }
        val n = (t - 1) * hop + nFft
        val y = DoubleArray(n); val env = DoubleArray(n)
        val re = DoubleArray(nFft); val im = DoubleArray(nFft)
        val fr = DoubleArray(nFft)
        for (f in 0 until t) {
            val base = f * 2 * nb
            for (k in 0 until nb) {
                val mag = minOf(Math.exp(rows[base + k].toDouble()), 1e2)
                val ph = rows[base + nb + k].toDouble()
                re[k] = mag * Math.cos(ph); im[k] = mag * Math.sin(ph)
            }
            im[0] = 0.0; im[nFft / 2] = 0.0                     // c2r ignores the imaginary part of DC and Nyquist
            for (k in 1 until nFft / 2) { re[nFft - k] = re[k]; im[nFft - k] = -im[k] }
            fft.inverseReal(re, im, fr)
            val o = f * hop
            for (i in 0 until nFft) { y[o + i] += fr[i] * win[i]; env[o + i] += win[i] * win[i] }
        }
        val pad = (nFft - hop) / 2
        return FloatArray(n - 2 * pad) { (y[pad + it] / maxOf(env[pad + it], 1e-11)).toFloat() }
    }
}

/** Mixed-radix complex FFT (factors 2, 3, 5, else a direct DFT of the remaining size). */
class Fft(private val n: Int) {
    private val wr = DoubleArray(n) { Math.cos(2 * Math.PI * it / n) }
    private val wi = DoubleArray(n) { Math.sin(2 * Math.PI * it / n) }     // + sign: inverse transform
    private val oRe = DoubleArray(n); private val oIm = DoubleArray(n)
    private val tr = DoubleArray(8); private val ti = DoubleArray(8)

    /** out[t] = (1/n) * Re(sum_k X[k] e^{+2 pi i k t / n}). */
    @Synchronized
    fun inverseReal(re: DoubleArray, im: DoubleArray, out: DoubleArray) {
        rec(re, im, 0, 1, n, 0)
        for (i in 0 until n) out[i] = oRe[i] / n
    }

    private fun factor(m: Int): Int = when { m % 2 == 0 -> 2; m % 3 == 0 -> 3; m % 5 == 0 -> 5; else -> m }

    private fun rec(xr: DoubleArray, xi: DoubleArray, off: Int, stride: Int, m: Int, outOff: Int) {
        if (m == 1) { oRe[outOff] = xr[off]; oIm[outOff] = xi[off]; return }
        val p = factor(m); val q = m / p
        for (r in 0 until p) rec(xr, xi, off + r * stride, stride * p, q, outOff + r * q)
        val step = n / m
        val ar = if (p <= 8) tr else DoubleArray(p); val ai = if (p <= 8) ti else DoubleArray(p)
        for (k in 0 until q) {
            for (r in 0 until p) { ar[r] = oRe[outOff + r * q + k]; ai[r] = oIm[outOff + r * q + k] }
            for (s in 0 until p) {
                var sr = 0.0; var si = 0.0
                val kk = k + s * q
                for (r in 0 until p) {
                    val e = ((r.toLong() * kk * step) % n).toInt()
                    val c = wr[e]; val d = wi[e]
                    sr += ar[r] * c - ai[r] * d; si += ar[r] * d + ai[r] * c
                }
                oRe[outOff + kk] = sr; oIm[outOff + kk] = si
            }
        }
    }
}
