#!/usr/bin/env python3
"""DviTTS-Nano on ONNX Runtime: the reference runner for phones, laptops and any non-PyTorch port.

Needs only numpy, onnxruntime and tokenizers (+ num2words and indic-numtowords for the text front end). No torch.
This file IS the spec: a port (Kotlin, Swift, C++) does exactly these steps with the files in the model directory.

  tts = DviTTS("dvitts-nano/")                    # model dir: config.json, *.onnx, tokenizer.json, text_map.json, voices/
  y, sr = tts.say("नमस्ते, आप कैसे हैं?", voice="ananya")
  tts.voices()                                     # preset names

Pipeline (stage 1 ONNX, MeanFlow 1 step, decoder, edge trim, AudioSeal watermark):
  1 front end   frontend.normalize (numbers, dates, symbols spelled out) and frontend.split for long text
  2 tokens      Qwen BPE (tokenizer.json) on " " + text; prefix = instruct + voice transcript + text, then the text map
  3 stage 1     autoregressive speech ids (25 Hz) with a KV cache; RAS sampler (temp .8, top-k 25, top-p .8, repeat guard);
                end-of-speech banned before min_ratio x text tokens; cap from the text length; one redraw if no stop
  4 MeanFlow    one step: acoustic digits (50 Hz, 6 dims x 6 levels) from content ids + the voice's prompt frames
  5 decoder     content + acoustic ids -> log-magnitude/phase rows -> iSTFT (n_fft 1920, hop 480) -> 24 kHz audio
  6 trim        Silero VAD: dead air at the start and end cut (0.1 s pad kept)
  7 watermark   AudioSeal 16-bit mark, added in 2 s chunks (EU AI Act Art. 50(2); on by default)
Randomness: two streams per call, both from the seed. `rng_factory(seed)` gives an object with .uniform() (stage 1, one
draw per sampled id) and .normal(shape) (MeanFlow noise); the default is numpy's PCG64. Same seed -> same audio."""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend as fe  # noqa: E402  (text front end, shipped next to this file)

# ---- script runs, length rules (same as DviTTS) -------------------------------------------------------------------
_SCRIPT = [("deva", 0x0900, 0x097F), ("beng", 0x0980, 0x09FF), ("guru", 0x0A00, 0x0A7F), ("gujr", 0x0A80, 0x0AFF),
           ("orya", 0x0B00, 0x0B7F)]
_LANG_OF = {"latin": "en", "deva": "hi", "beng": "bn", "guru": "pa", "gujr": "gu", "orya": "or"}
SPOKEN = "English, Hindi, Marathi, Bengali, Punjabi, Gujarati, Odia"


def runs(text: str) -> list[tuple[str, str]]:
    """Text -> [(script, run)]: Latin letters vs each Indic block; digits/punctuation/space join the current run."""
    out: list[tuple[str, str]] = []
    for ch in text:
        o = ord(ch)
        s = next((n for n, a, b in _SCRIPT if a <= o <= b), "latin" if ch.isalpha() else None)
        if out and (s is None or s == out[-1][0]):
            out[-1] = (out[-1][0], out[-1][1] + ch)
        elif s is not None:
            out.append((s, ch))
        elif not out:
            out.append(("latin", ch))
    return out


def unsupported_scripts(text: str) -> list[str]:
    bad = set()
    for ch in text:
        o = ord(ch)
        if not ch.isalpha() or o <= 0x024F or any(a <= o <= b for _, a, b in _SCRIPT) or 0x1E00 <= o <= 0x1EFF:
            continue
        bad.add(unicodedata.name(ch, f"U+{o:04X}").split()[0])
    return sorted(bad)


def normalize(text: str, lang: str) -> str:
    """frontend.normalize on its portable path: English without NeMo (num2words rules + _en_pre), as every port does.
    NeMo is blocked for the call even if installed, so this file and the Kotlin/Swift ports give the same text."""
    had = "nemo_text_processing" in sys.modules
    old = sys.modules.get("nemo_text_processing")
    sys.modules["nemo_text_processing"] = None                 # import -> ImportError -> fallback rules
    try:
        return fe.normalize(text, lang)
    finally:
        if had:
            sys.modules["nemo_text_processing"] = old
        else:
            sys.modules.pop("nemo_text_processing", None)


def guess_lang(text: str) -> str:
    n: dict[str, int] = {}
    for s, r in runs(text):
        n[s] = n.get(s, 0) + sum(ch.isalpha() or 0x0900 <= ord(ch) <= 0x0B7F for ch in r)
    return _LANG_OF[max(n, key=n.get)] if n else "en"


class NpRng:
    """Default randomness: numpy PCG64 from the seed. Ports may use any good generator (the audio then differs from
    this file's for the same seed, never in kind)."""

    def __init__(self, seed: int):
        self.g = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def uniform(self) -> float:
        return float(self.g.random())

    def normal(self, shape) -> np.ndarray:
        return self.g.standard_normal(shape).astype(np.float32)


class DviTTS:
    def __init__(self, model_dir: str | os.PathLike, threads: int = 4, watermark: bool = True, trim: bool = True,
                 rng_factory=NpRng, spin: bool | None = None, laptop: bool | None = None, s1_gguf: str | None = None,
                 wm_file: str | None = None, low_mem: bool | str | None = None, s1_ctx: int = 2048):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.dir = Path(model_dir)
        self.cfg = c = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.rng_factory = rng_factory
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        # Thread spinning (speed only; output is bit-identical either way). Off by default: on a phone it stole cores, and
        # on a laptop it gave no clear win; spin=True to try.
        if spin is None:
            spin = False
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
        # Memory. ONNX Runtime's CPU memory pool (arena) keeps every block it ever grabbed, and memory pattern planning
        # reserves scratch space per input size; with sentences of different lengths both only grow.
        #   "all"  (True): pool + pattern off. Lowest memory; no speed cost on Linux or Android, ~15 % slower on Windows.
        #   "shared": one pool for all sessions. Same speed everywhere, most of the saving.
        #   "off"  (False): ONNX Runtime defaults. Also "nopattern", "shared_grow", "shared_nopattern".
        #   None (default): "shared" on x86 and macOS, "all" on other ARM (phones). Phone peak 1125 -> 865 MB,
        #   Windows laptop 927 -> 770 MB, same speed; M1 Pro "shared" vs "all": 0.27 vs 0.30 x real time, lower peak.
        # Audio is bit-identical in every mode.
        import platform
        if low_mem is None:
            x86_or_mac = platform.machine().lower() in ("x86_64", "amd64", "i686", "x86") or sys.platform == "darwin"
            low_mem = "shared" if x86_or_mac else "all"
        low_mem = {True: "all", False: "off"}.get(low_mem, low_mem)
        if low_mem not in ("all", "off", "nopattern", "shared", "shared_grow", "shared_nopattern"):
            raise ValueError(f"low_mem: {low_mem!r}")
        self.low_mem = low_mem
        if low_mem == "all":
            so.enable_cpu_mem_arena = False
        if low_mem in ("all", "nopattern", "shared_nopattern"):
            so.enable_mem_pattern = False
        if low_mem in ("shared", "shared_grow", "shared_nopattern"):
            mi = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            try:  # one shared allocator per process; a second DviTTS in the same process reuses the first one
                ort.create_and_register_allocator(mi, ort.OrtArenaCfg(0, 1 if low_mem == "shared_grow" else -1, -1, -1))
            except Exception:
                pass
            so.add_session_config_entry("session.use_env_allocators", "1")
        # s1_ctx: llama.cpp context of the laptop stage 1. One chunk = prompt + text + <= max_s * 25 ids, well under 2048;
        # 4096 -> 2048 saves ~13 MB at the same speed, audio identical (c7i, 3 rounds).
        self.s1_ctx = int(s1_ctx)
        sess = lambda f: ort.InferenceSession(str(self.dir / f), so, providers=["CPUExecutionProvider"])  # noqa: E731
        f = c["files"]
        # Laptop path: on x86 (laptops, desktops) stage 1 runs on llama.cpp from files.s1_gguf (a GGUF
        # dump of the same stage 1; pip install llama-cpp-python) and the watermark is files.wm_x86 (fp32: ONNX Runtime runs
        # the 8-bit ConvInteger graph slowly on x86). Ryzen 7 7735HS, Windows, 8 threads: 0.98 -> 0.63 x real time; same
        # quality as the ONNX stage 1 (FLEURS hi, IV, Seed EN) but not bit-identical to it (different 8-bit format).
        # laptop: None = auto (x86 + llama_cpp installed + both files present), True = require, False = phone path.
        # s1_gguf / wm_file name the files directly and win over laptop.
        import platform
        x86 = platform.machine().lower() in ("x86_64", "amd64", "i686", "x86")
        if laptop is None:
            try:
                import llama_cpp  # noqa: F401
                has_llama = True
            except ImportError:
                has_llama = False
            have = all(f.get(k) and (self.dir / f[k]).exists() for k in ("s1_gguf", "wm_x86"))
            laptop = x86 and has_llama and have
            if x86 and have and not has_llama:
                # No llama.cpp: stage 1 stays on ONNX, but the fp32 watermark still helps on x86 (7735HS, 8 threads:
                # 0.98 -> 0.81 x real time). Say so once, so a slow laptop is not a silent mystery.
                import warnings
                warnings.warn("DviTTS: llama-cpp-python is not installed, so stage 1 runs on ONNX Runtime (about 1.3x "
                              "slower on x86 laptops). For the laptop speed-up: pip install --no-deps -r "
                              "requirements-laptop.txt && pip install typing-extensions diskcache jinja2 (see README).",
                              RuntimeWarning, stacklevel=2)
                wm_file = wm_file or f["wm_x86"]
        if laptop:
            if not (f.get("s1_gguf") and f.get("wm_x86")):
                raise ValueError("laptop=True needs files.s1_gguf and files.wm_x86 in config.json")
            s1_gguf = s1_gguf or f["s1_gguf"]
            wm_file = wm_file or f["wm_x86"]
        self.path = "laptop" if s1_gguf else ("phone+wm_x86" if wm_file and wm_file == f.get("wm_x86") else "phone")
        self.llm = None
        if s1_gguf:
            from llama_cpp import Llama
            gp = Path(s1_gguf) if Path(s1_gguf).is_absolute() else self.dir / s1_gguf
            self.llm = Llama(model_path=str(gp), n_ctx=self.s1_ctx, n_threads=threads, n_threads_batch=threads, n_gpu_layers=0,
                             logits_all=False, verbose=False)
            self.s1 = None
        else:
            self.s1 = sess(f["s1"])
        self.mf, self.dec = sess(f["mf"]), sess(f["dec"])
        self.wm = sess(wm_file or f["wm"]) if watermark else None
        vso = ort.SessionOptions()
        vso.intra_op_num_threads = vso.inter_op_num_threads = 1
        self.vad = ort.InferenceSession(str(self.dir / f["vad"]), vso, providers=["CPUExecutionProvider"]) if trim else None
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        tm = json.loads((self.dir / "text_map.json").read_text(encoding="utf-8"))
        self.tm_new = {int(o): n for n, o in enumerate(tm["keep"])}
        self.tm_fb = {int(k): [self.tm_new[int(j)] for j in v] for k, v in tm["fb"].items()}
        k = np.load(self.dir / "consts.npz")
        self.vals, self.basis = k["vals"].astype(np.float32), k["basis"].astype(np.int64)
        self.rs_kernel, self.rs_width = k["rs_kernel"].astype(np.float32), int(k["rs_width"])   # 24k -> 16k (VAD)
        self.instr = [int(i) for i in c["instruct_ids"]]

    # ---- voices -----------------------------------------------------------------------------------------------------
    def voices(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "voices").glob("*.npz"))

    def match_voice(self, name: str, lang: str) -> str:
        """A preset speaks its own side of the English / Indian split only (English text in an Indian
        voice comes out silent or short, and the reverse too; Indian voices read every Indian language). If the text's
        language is on the other side, return the first preset of that language with the same gender (Hindi if the
        language has none), else the voice unchanged."""
        idx = self.index()
        m = idx.get(name)
        if m is None or (lang == "en") == (m["lang"] == "en"):
            return name
        pool = [k for k in sorted(idx) if idx[k]["lang"] == lang] or [k for k in sorted(idx) if idx[k]["lang"] == "hi"]
        same = [k for k in pool if idx[k].get("gender") == m.get("gender")]
        return (same or pool or [name])[0]

    def plan_xling(self, voice: str | dict, lang: str, match: bool, xling: bool, info: dict | None = None):
        """Keep the pick on English/Indian mismatch; drop S1 prompt. -> (xling, voice). Writes info['xling']."""
        info = {} if info is None else info
        if xling:
            info["xling"] = True
            return True, voice
        if match and isinstance(voice, str):
            to = self.match_voice(voice, lang)
            if to != voice:
                info["xling"] = True
                return True, voice
        return False, voice

    def index(self) -> dict:
        if not hasattr(self, "_index"):
            f = self.dir / "voices" / "index.json"
            self._index = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return self._index

    def voice(self, name: str) -> dict:
        z = np.load(self.dir / "voices" / f"{name}.npz")
        return {k: z[k] for k in z.files}

    # ---- text -------------------------------------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Qwen BPE ids (before the text map). The stack encodes " " + text for every text it tokenizes."""
        return self.tok.encode(text, add_special_tokens=bool(self.cfg.get("tok_add_special", False))).ids

    def text_map(self, ids) -> list[int]:
        out: list[int] = []
        for i in ids:
            n = self.tm_new.get(int(i))
            out.extend([n] if n is not None else self.tm_fb[int(i)])
        return out

    def exp_ids(self, text: str) -> float:
        ts = self.cfg["tok_s"]
        return 25 * sum(len(self.encode(" " + r)) / ts.get(n, 2.8) for n, r in runs(text))

    def min_ratio(self, text: str, n_text: int) -> float:
        c = self.cfg
        rt = [(n, len(self.encode(" " + r))) for n, r in runs(text)]
        if all(c["min_scale"].get(s, 1.0) >= 1.0 for s, _ in rt) or n_text <= 0:
            return c["min_ratio"]
        eff = sum(n * min(1.0, c["min_scale"].get(s, 1.0)) for s, n in rt)
        return c["min_ratio"] * min(1.0, eff / n_text)

    # ---- public -----------------------------------------------------------------------------------------------------
    def say(self, text: str, voice: str | dict, seed: int = 0, lang: str | None = None, normalize: bool = True,
            info: dict | None = None, match: bool = True, xling: bool = False) -> tuple[np.ndarray, int]:
        """Text -> (float32 mono audio, 24000). Long text is split into chunks of <= ~25 s, joined with 0.2 s silence.
        match: English vs Indian split (match_voice). On mismatch, keep the pick and drop the stage-1 prompt;
        MeanFlow still uses that pick's frames. info["xling"]=True. match=False leaves a cross-lang prompt in (plain;
        often silent). Explicit xling=True does the same drop without consulting match.
        Voice identity on that path is weak."""
        c = self.cfg
        bad = unsupported_scripts(text)
        if bad:
            raise ValueError(f"Unsupported script: {', '.join(bad)}. This model speaks {SPOKEN}.")
        lang = lang or guess_lang(text)
        info = {} if info is None else info
        xling, voice = self.plan_xling(voice, lang, match, xling, info)
        v = self.voice(voice) if isinstance(voice, str) else voice
        if normalize:
            text = globals()["normalize"](text, lang)
        ic = int(c.get("in_chunk") or 0) if lang != "en" else 0      # Indian text in chunks of <= in_chunk chars
        if (len(text) > fe.MAX_CHARS.get(lang, fe.MAX_CHARS_IN) or c["rate_scale"] * self.exp_ids(text) / 25 > c["split_s"]
                or (ic and len(text) > ic)):
            chunks = []
            for ch in (fe.split(text, lang, max_chars=ic, min_chars=int(ic * .6)) if ic else fe.split(text, lang)):
                e = c["rate_scale"] * self.exp_ids(ch) / 25
                if e <= c["split_s"]:
                    chunks.append(ch)
                    continue
                mx = max(40, int(len(ch) / -(-e // c["split_s"])))
                chunks += fe.split(ch, lang, max_chars=mx, min_chars=int(mx * .6))
            info["chunks"], parts = [], []
            gap = np.zeros(int(c["gap_s"] * c["sr"]), np.float32)
            for i, ch in enumerate(chunks):
                ci = {"text": ch}
                y = self._say_one(v, ch, seed * 1000 + i, ci, xling=xling)
                info["chunks"].append(ci)
                parts += [y, gap] if i < len(chunks) - 1 else [y]
            y = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            info["n_ids"] = sum(ci.get("n_ids", 0) for ci in info["chunks"])     # as TTSStack.say on its long route
            info["eos"] = all(ci.get("eos", True) for ci in info["chunks"])
        else:
            y = self._say_one(v, text, seed, info, xling=xling)
        if self.wm is not None:
            y = self.mark(y)
        return y.astype(np.float32), c["sr"]

    # ---- one chunk --------------------------------------------------------------------------------------------------
    def _say_one(self, v: dict, text: str, seed: int, info: dict, xling: bool = False, gap_retry: bool = True) -> np.ndarray:
        """One chunk. With trim on, a take whose longest non-speech stretch inside the clip is >= gap_retry_s is made
        once more on seed + gap_seed; the take with the shorter inside gap is kept (tie: first). About 5 %
        of Punjabi takes had 2.4-4.0 s gaps of low sound (not silence); a new seed did not repeat them."""
        c = self.cfg
        tt = self.encode(" " + text)
        exp = self.exp_ids(text)
        cap = min(int(c["cap_x"] * exp), int(c["max_s"] * 25))
        mr = self.min_ratio(text, len(tt))
        ref_text = [] if xling else [int(i) for i in v["ref_text_ids"]]
        ref_ids = [] if xling else v["ref_ids"]
        prefix = self.text_map(self.instr + ref_text + tt)
        ids, eos = self.stage1(prefix, ref_ids, len(tt), self.rng_factory(seed), mr, cap)
        info["redraw"] = 0
        for r in range(1, c["redraws"] + 1):                     # runaway guard: draw again if it never stopped
            if eos:
                break
            ids2, eos2 = self.stage1(prefix, ref_ids, len(tt), self.rng_factory(seed + r * c["redraw_seed"]), mr, cap)
            info["redraw"] = r
            if eos2:
                ids, eos = ids2, eos2
        info.update(n_ids=len(ids), eos=eos, exp_ids=exp, cap=cap, min_ratio=mr, ids=ids)
        n24 = len(ids) * (c["sr"] // 25)
        if len(ids) < 5:
            return np.zeros(n24, np.float32)
        a_hat = self.meanflow(v, ids, self.rng_factory(seed))
        y = self.decode(ids, a_hat, n24)
        if self.vad is None:
            return y
        dur, lead, tail, gap = self._bounds(y)
        sr, pad = c["sr"], c["trim_pad"]
        y = y[max(0, int((lead - pad) * sr)):min(len(y), int((dur - tail + pad) * sr))]
        info["gap"], info["gap_retry"] = gap, 0
        gr = float(c.get("gap_retry_s") or 0)
        if gap_retry and gr > 0 and gap >= gr:
            i2: dict = {}
            y2 = self._say_one(v, text, seed + int(c["gap_seed"]), i2, xling=xling, gap_retry=False)
            if i2["gap"] < gap:
                info.update(i2)
                y = y2
            info["gap_retry"], info["gap_first"] = 1, gap
        return y

    # ---- stage 1 ----------------------------------------------------------------------------------------------------
    def stage1(self, text_ids: list[int], ref_ids, n_text: int, rng, min_ratio: float, cap: int):
        """-> (speech ids, stopped). Model ids: text < offset, speech = offset + id; logits are the speech rows only."""
        c = self.cfg
        off, nsp, eos_id = c["offset"], c["n_speech"], c["eos"]
        m = c["s1_shape"]
        prefix = [off + c["sos"]] + list(text_ids) + [off + c["task"]] + [off + int(i) for i in ref_ids]
        lo, hi = int(n_text * min_ratio), max(int(n_text * c["max_ratio"]), 1)
        hi = min(hi, max(cap, lo + 1))
        pk = pv = np.zeros((m["L"], m["KV"], 0, m["hd"]), np.float32)
        feed = np.asarray([prefix], np.int64)
        out: list[int] = []
        if self.llm is not None:
            self.llm.reset()
            if len(prefix) + hi > self.s1_ctx:                   # never run past the llama.cpp context
                hi = max(lo + 1, self.s1_ctx - len(prefix))
        for t in range(hi):
            if self.llm is not None:                             # GGUF vocab = text rows, then the speech rows at offset
                self.llm.eval([int(i) for i in feed[0]])
                V = int(self.llm.n_vocab())
                z = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(V,))[off:].astype(np.float32)
            else:
                lg, pk, pv = self.s1.run(None, {"ids": feed, "past_k": pk, "past_v": pv})
                z = lg[0, -1].astype(np.float32)
            z[nsp + 2:] = -np.inf
            z[nsp] = -np.inf                                     # SOS row
            if t < lo:
                z[eos_id] = -np.inf
            j = self.sample(z, out, rng)
            if j == eos_id:
                return out, True
            out.append(j)
            feed = np.asarray([[off + j]], np.int64)
        return out, False

    def sample(self, logits: np.ndarray, recent: list[int], rng) -> int:
        """RAS (VALL-E 2): softmax(logits / temp), top-k by probability, nucleus top-p inside them (first always kept),
        inverse-CDF draw; if the id occurs >= max(1, win * tau_r) times in the last win ids, redraw from the full
        distribution."""
        s = self.cfg["sampler"]
        z = logits.astype(np.float64) / s["temp"]
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        k = min(s["top_k"], len(p))
        idx = np.argpartition(-p, k - 1)[:k]
        idx = idx[np.argsort(-p[idx], kind="stable")]
        vals = p[idx]
        keep = (np.cumsum(vals) - vals) < s["top_p"]
        keep[0] = True
        cq = np.cumsum(vals * keep)
        u = rng.uniform() * cq[-1]
        j = int(idx[min(int(np.searchsorted(cq, u, side="right")), k - 1)])
        win = s["win"]
        if recent and sum(r == j for r in recent[-win:]) >= max(1, int(win * s["tau_r"])):
            cp = np.cumsum(p)
            u = rng.uniform() * cp[-1]
            j = min(int(np.searchsorted(cp, u, side="right")), len(p) - 1)
        return j

    # ---- MeanFlow + decoder -----------------------------------------------------------------------------------------
    def meanflow(self, v: dict, ids: list[int], rng) -> np.ndarray:
        """Acoustic ids at 50 Hz: one MeanFlow step from noise, prompt frames as context, snapped to the 6 levels."""
        c_t = np.repeat(np.asarray(ids, np.int64), 2)
        c_r, ac_r = v["c_r"].astype(np.int64), v["ac_r"].astype(np.int64)
        dig_ref = (ac_r[:, None] // self.basis) % len(self.vals)
        x_ref = self.vals[dig_ref]
        x = rng.normal((len(c_t), len(self.basis))) * self.cfg["s2_noise"]
        content = np.concatenate([c_r, c_t])[None]
        seg = np.concatenate([np.zeros(len(c_r), np.int64), np.ones(len(c_t), np.int64)])[None]
        xv = np.concatenate([x_ref, x]).astype(np.float32)[None]
        u = self.mf.run(["u"], {"content": content, "xv": xv, "seg": seg})[0][0]
        x1 = x + u[len(c_r):]
        dig = np.abs(x1[..., None] - self.vals).argmin(-1)
        return (dig.astype(np.int64) * self.basis).sum(-1)

    def decode(self, ids, a_hat: np.ndarray, n24: int) -> np.ndarray:
        rows = self.dec.run(["logits"], {"idx_s": np.asarray(ids, np.int64)[None], "idx_a": a_hat.astype(np.int64)[None]})[0][0]
        y = istft(rows, self.cfg["n_fft"], self.cfg["hop"])
        return (y[:n24] if len(y) >= n24 else np.pad(y, (0, n24 - len(y)))).astype(np.float32)

    # ---- trim + watermark -------------------------------------------------------------------------------------------
    def resample_24_16(self, y: np.ndarray) -> np.ndarray:
        """torchaudio's sinc resample (24 -> 16 kHz), fixed kernel from consts.npz."""
        o, n, w = 3, 2, self.rs_width
        L = len(y)
        yp = np.pad(y.astype(np.float32), (w, w + o))
        kern = self.rs_kernel.reshape(n, -1)                     # [n, K]
        K = kern.shape[1]
        nfr = (len(yp) - K) // o + 1
        fr = np.lib.stride_tricks.as_strided(yp, (nfr, K), (yp.strides[0] * o, yp.strides[0]))
        return (fr @ kern.T).reshape(-1)[: -(-L * n // o)]

    def speech_bounds(self, y24: np.ndarray) -> tuple[float, float, float]:
        """Silero VAD on 16 kHz (= silero_vad.get_speech_timestamps with min_silence 300 ms, pad 30 ms, seconds rounded
        to 0.1) -> (duration, lead, tail) in seconds."""
        return self._bounds(y24)[:3]

    def _bounds(self, y24: np.ndarray) -> tuple[float, float, float, float]:
        """speech_bounds + the longest gap between speech spans."""
        x = self.resample_24_16(y24)
        sr, ws, ctx_n = 16000, 512, 64
        st = np.zeros((2, 1, 128), np.float32)
        ctx = np.zeros((1, ctx_n), np.float32)
        probs = []
        for i in range(0, len(x), ws):
            ch = x[i:i + ws]
            ch = np.pad(ch, (0, ws - len(ch)))[None].astype(np.float32)
            inp = np.concatenate([ctx, ch], 1)
            out, st = self.vad.run(None, {"input": inp, "state": st, "sr": np.array(sr, np.int64)})
            ctx = inp[:, -ctx_n:]
            probs.append(float(out.reshape(-1)[0]))
        thr, neg = 0.5, 0.35
        min_sp, pad, min_sil = sr * 0.25, sr * 0.03, sr * 0.3
        n = len(x)
        trig, temp_end, cur, sp = False, 0, {}, []
        for i, pr in enumerate(probs):
            s = ws * i
            if pr >= thr and temp_end:
                temp_end = 0
            if pr >= thr and not trig:
                trig, cur = True, {"start": s}
                continue
            if pr < neg and trig:
                if not temp_end:
                    temp_end = s
                if s - temp_end < min_sil:
                    continue
                cur["end"] = temp_end
                if cur["end"] - cur["start"] > min_sp:
                    sp.append(cur)
                cur, temp_end, trig = {}, 0, False
        if cur and n - cur["start"] > min_sp:
            cur["end"] = n
            sp.append(cur)
        for i, s_ in enumerate(sp):
            if i == 0:
                s_["start"] = int(max(0, s_["start"] - pad))
            if i != len(sp) - 1:
                gap = sp[i + 1]["start"] - s_["end"]
                if gap < 2 * pad:
                    s_["end"] += int(gap // 2)
                    sp[i + 1]["start"] = int(max(0, sp[i + 1]["start"] - gap // 2))
                else:
                    s_["end"] = int(min(n, s_["end"] + pad))
                    sp[i + 1]["start"] = int(max(0, sp[i + 1]["start"] - pad))
            else:
                s_["end"] = int(min(n, s_["end"] + pad))
        dur = n / sr
        if not sp:
            return round(dur, 2), round(dur, 2), 0.0, 0.0
        start = max(round(sp[0]["start"] / sr, 1), 0)
        end = min(round(sp[-1]["end"] / sr, 1), dur)
        gap = max((round(b["start"] / sr, 1) - round(a["end"] / sr, 1) for a, b in zip(sp, sp[1:])), default=0.0)
        return round(dur, 2), round(start, 2), round(dur - end, 2), round(gap, 2)

    def trim(self, y: np.ndarray) -> np.ndarray:
        sr, pad = self.cfg["sr"], self.cfg["trim_pad"]
        dur, lead, tail = self.speech_bounds(y)
        return y[max(0, int((lead - pad) * sr)):min(len(y), int((dur - tail + pad) * sr))]

    def mark(self, y: np.ndarray) -> np.ndarray:
        """AudioSeal mark: fixed 2 s graph, the clip in 2 s chunks (last zero-padded), cut back to length."""
        if len(y) == 0:
            return y
        ch = self.cfg["wm_chunk"]
        yp = np.pad(y.astype(np.float32), (0, -len(y) % ch))
        out = [self.wm.run(None, {"y": yp[i:i + ch][None]})[0][0] for i in range(0, len(yp), ch)]
        return np.concatenate(out)[: len(y)]


def istft(rows: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Vocos ISTFT head, 'same' padding: rows [T, 2 * (n_fft // 2 + 1)] = log-magnitude | phase."""
    logm, ph = np.split(rows.astype(np.float64), 2, axis=-1)
    fr = np.fft.irfft(np.minimum(np.exp(logm), 1e2) * np.exp(1j * ph), n=n_fft, axis=-1)
    win = np.hanning(n_fft + 1)[:-1]
    fr *= win
    n = (len(fr) - 1) * hop + n_fft
    y, env = np.zeros(n), np.zeros(n)
    for i, f in enumerate(fr):
        y[i * hop:i * hop + n_fft] += f
        env[i * hop:i * hop + n_fft] += win ** 2
    pad = (n_fft - hop) // 2
    y, env = y[pad:n - pad], env[pad:n - pad]
    return (y / np.maximum(env, 1e-11)).astype(np.float32)


if __name__ == "__main__":
    import argparse
    import wave
    ap = argparse.ArgumentParser(description="DviTTS ONNX runner")
    ap.add_argument("model_dir")
    ap.add_argument("text")
    ap.add_argument("--voice", default="")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--spin", choices=("auto", "on", "off"), default="auto", help="ONNX Runtime thread spinning (speed only; auto = off)")
    ap.add_argument("--path", choices=("auto", "phone", "laptop"), default="auto",
                    help="auto = laptop path on x86 when llama-cpp-python is installed, else the phone path")
    ap.add_argument("--s1-gguf", default="", help="stage 1 on llama.cpp from this GGUF (laptop path; needs llama-cpp-python)")
    ap.add_argument("--wm", default="", help="watermark graph to use instead of config files.wm (e.g. wm_fp32.onnx)")
    a = ap.parse_args()
    t = DviTTS(a.model_dir, threads=a.threads, spin={"auto": None, "on": True, "off": False}[a.spin],
               laptop={"auto": None, "phone": False, "laptop": True}[a.path], s1_gguf=a.s1_gguf or None, wm_file=a.wm or None)
    info: dict = {}
    y, sr = t.say(a.text, a.voice or t.voices()[0], seed=a.seed, info=info)
    if info.get("xling"):
        print("cross-lang: kept the picked voice; dropped the S1 prompt (MeanFlow still uses its frames)")
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"{a.out}: {len(y) / sr:.2f} s")
