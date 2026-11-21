#!/usr/bin/env python3
"""DviTTS text front end: written form -> spoken form, then sentence chunks.
The judged evals fed one clean sentence at a time; real input has numbers, dates, money, URLs and whole paragraphs.

normalize(text, lang)
  en          NeMo text processing (WFST, Apache 2.0; numbers, dates, money, times, units, URLs, abbreviations). Loaded once,
              lazily; falls back to num2words rules if nemo_text_processing is missing. NeMo's Hindi grammar was tried and
              dropped ("₹1,200" -> "एक रुपए,दो सौ"; whole sentences returned unnormalised).
  hi mr gu bn pa or   own rules below + AI4Bharat indic-numtowords (MIT) for the number words: native digits -> ASCII, URLs and
              e-mails spelled, dd/mm/yyyy dates, hh:mm times, money (₹ Rs INR $), percent, phone numbers digit by digit,
              decimals, Hindi years 1100-1999 as "उन्नीस सौ सड़सठ", grouped numbers (1,200 and 1,20,000), plain numbers.
              Latin-script English words are left as they are.
split(text, lang)  CosyVoice split_paragraph shape: cut at sentence ends (and newlines), cut long sentences at commas then
              words, merge short pieces up to the limit. Limits are characters (no tokenizer needed here).
Only the number words, month names and a few symbol words are language data; everything else is shared."""
from __future__ import annotations

import datetime
import re
import unicodedata

LANGS = ("en", "hi", "mr", "gu", "bn", "pa", "or")
# One chunk: up to about 25 s of speech (stage 1 trained on 1.5-30 s clips). Shorter chunks (240 / 200 characters)
# sounded slow and robotic in listening tests (20-50 % slower than one unsplit pass); one pass stays clean up to
# ~650 characters (en) and ~500 (hi).
MAX_CHARS = {"en": 500}
MAX_CHARS_IN = 400
MIN_CHARS = {"en": 300}        # merge target; fewer cuts = fewer prosody resets
MIN_CHARS_IN = 250

_DIGITS = {0x0966: "hi", 0x09E6: "bn", 0x0A66: "pa", 0x0AE6: "gu", 0x0B66: "or"}   # zero code point of each script
_DIGIT_MAP = {chr(z + k): str(k) for z in _DIGITS for k in range(10)}

# symbol and unit words per language (Devanagari words for hi and mr; the rest in their own scripts)
WORDS = {
    "hi": {"rupee": "रुपये", "dollar": "डॉलर", "percent": "प्रतिशत", "point": "दशमलव", "dot": "डॉट", "at": "एट",
           "slash": "स्लैश", "plus": "प्लस", "and": "और", "baje": "बजे", "bajkar": "बजकर", "minute": "मिनट", "hundred": "सौ"},
    "mr": {"rupee": "रुपये", "dollar": "डॉलर", "percent": "टक्के", "point": "पूर्णांक", "dot": "डॉट", "at": "एट",
           "slash": "स्लॅश", "plus": "प्लस", "and": "आणि", "minute": "मिनिटे"},
    "gu": {"rupee": "રૂપિયા", "dollar": "ડોલર", "percent": "ટકા", "point": "દશાંશ", "dot": "ડોટ", "at": "એટ",
           "slash": "સ્લેશ", "plus": "પ્લસ", "and": "અને", "minute": "મિનિટ"},
    "bn": {"rupee": "টাকা", "dollar": "ডলার", "percent": "শতাংশ", "point": "দশমিক", "dot": "ডট", "at": "অ্যাট",
           "slash": "স্ল্যাশ", "plus": "প্লাস", "and": "এবং", "minute": "মিনিট"},
    "pa": {"rupee": "ਰੁਪਏ", "dollar": "ਡਾਲਰ", "percent": "ਪ੍ਰਤੀਸ਼ਤ", "point": "ਦਸ਼ਮਲਵ", "dot": "ਡੌਟ", "at": "ਐਟ",
           "slash": "ਸਲੈਸ਼", "plus": "ਪਲੱਸ", "and": "ਅਤੇ", "minute": "ਮਿੰਟ"},
    "or": {"rupee": "ଟଙ୍କା", "dollar": "ଡଲାର", "percent": "ପ୍ରତିଶତ", "point": "ଦଶମିକ", "dot": "ଡଟ", "at": "ଆଟ",
           "slash": "ସ୍ଲାସ", "plus": "ପ୍ଲସ", "and": "ଓ", "minute": "ମିନିଟ"},
}
MONTHS = {
    "hi": "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
    "mr": "जानेवारी फेब्रुवारी मार्च एप्रिल मे जून जुलै ऑगस्ट सप्टेंबर ऑक्टोबर नोव्हेंबर डिसेंबर".split(),
    "gu": "જાન્યુઆરી ફેબ્રુઆરી માર્ચ એપ્રિલ મે જૂન જુલાઈ ઓગસ્ટ સપ્ટેમ્બર ઓક્ટોબર નવેમ્બર ડિસેમ્બર".split(),
    "bn": "জানুয়ারি ফেব্রুয়ারি মার্চ এপ্রিল মে জুন জুলাই আগস্ট সেপ্টেম্বর অক্টোবর নভেম্বর ডিসেম্বর".split(),
    "pa": "ਜਨਵਰੀ ਫ਼ਰਵਰੀ ਮਾਰਚ ਅਪ੍ਰੈਲ ਮਈ ਜੂਨ ਜੁਲਾਈ ਅਗਸਤ ਸਤੰਬਰ ਅਕਤੂਬਰ ਨਵੰਬਰ ਦਸੰਬਰ".split(),
    "or": "ଜାନୁଆରୀ ଫେବୃଆରୀ ମାର୍ଚ୍ଚ ଅପ୍ରେଲ ମେ ଜୁନ ଜୁଲାଇ ଅଗଷ୍ଟ ସେପ୍ଟେମ୍ବର ଅକ୍ଟୋବର ନଭେମ୍ବର ଡିସେମ୍ବର".split(),
}
ABBR = {"hi": {"डॉ.": "डॉक्टर", "किमी": "किलोमीटर", "कि.मी.": "किलोमीटर", "सेमी": "सेंटीमीटर", "किग्रा": "किलोग्राम", "मि.ली.": "मिलीलीटर", "रु.": "रुपये"},
        "mr": {"डॉ.": "डॉक्टर", "किमी": "किलोमीटर"}}
CRORE = {"hi": "करोड़", "mr": "कोटी", "gu": "કરોડ", "bn": "কোটি", "pa": "ਕਰੋੜ", "or": "କୋଟି"}
_EN_WORDS = {"dot": "dot", "at": "at", "slash": "slash"}


def num_words(n: int | str, lang: str) -> str:
    """Cardinal words; en via num2words, Indic via indic-numtowords."""
    n = int(str(n))
    if lang == "en":
        from num2words import num2words
        return num2words(n, lang="en").replace(",", "").replace("-", " ")
    from indic_numtowords import num2words as inw
    if n >= 10 ** 7:                                  # indic-numtowords stops below 10^9: crores done here, recursively
        hi_, lo = divmod(n, 10 ** 7)
        return f"{num_words(hi_, lang)} {CRORE[lang]}" + (f" {num_words(lo, lang)}" if lo else "")
    return inw(n, lang=lang)


def digits_words(s: str, lang: str) -> str:
    return " ".join(num_words(int(c), lang) for c in s if c.isdigit())


def _w(lang: str, key: str) -> str:
    return WORDS.get(lang, {}).get(key) or _EN_WORDS.get(key, key)


def _year_hi(n: int) -> str:
    hi, lo = divmod(n, 100)
    return f"{num_words(hi, 'hi')} सौ" + (f" {num_words(lo, 'hi')}" if lo else "")


def _spell_net(s: str, lang: str) -> str:
    """URL or e-mail -> spoken pieces: letters kept, '.' '@' '/' as words."""
    s = re.sub(r"^https?://", "", s).rstrip("/")
    out = []
    for part in re.split(r"([./@])", s):
        if part == ".":
            out.append(_w(lang, "dot"))
        elif part == "/":
            out.append(_w(lang, "slash"))
        elif part == "@":
            out.append(_w(lang, "at"))
        elif part:
            out.append(re.sub(r"\d+", lambda m: " " + digits_words(m.group(), lang) + " ", part).strip())
    return " ".join(out)


_NUM = r"\d{1,3}(?:,\d{2,3})+|\d+"


def _indic(text: str, lang: str) -> str:
    t = text.translate(str.maketrans(_DIGIT_MAP))
    for a, b in ABBR.get(lang, {}).items():
        t = re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, t)
    t = re.sub(r"(https?://\S+|www\.\S+)", lambda m: _spell_net(m.group().rstrip(".,"), lang), t)
    if lang in ("hi", "mr"):
        t = re.sub(r"\bwww\b", "डब्ल्यू डब्ल्यू डब्ल्यू", t)
    t = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", lambda m: _spell_net(m.group(), lang), t)

    def date(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group()
        yw = _year_hi(int(y)) if lang == "hi" and 1100 <= int(y) <= 1999 else num_words(int(y), lang)
        return f"{num_words(d, lang)} {MONTHS[lang][mo - 1]} {yw}"
    t = _date8(t, lang)
    t = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", date, t)

    def clock(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group()
        if lang == "hi":
            return f"{num_words(h, lang)} {_w(lang, 'baje')}" if mi == 0 else \
                f"{num_words(h, lang)} {_w(lang, 'bajkar')} {num_words(mi, lang)} {_w(lang, 'minute')}"
        return f"{num_words(h, lang)} {num_words(mi, lang)}" if mi else num_words(h, lang)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b(?:\s*बजे)?", clock, t)

    def phone(m):
        s = m.group()
        lead = _w(lang, "plus") + " " if s.startswith("+") else ""
        return lead + digits_words(s, lang)
    t = re.sub(r"\+?\d[\d -]{8,}\d", lambda m: phone(m) if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), t)

    def dec(m):
        a, b = m.group(1).replace(",", ""), m.group(2)
        return f"{num_words(int(a), lang)} {_w(lang, 'point')} {digits_words(b, lang)}"
    money_num = rf"(?:{_NUM})(?:\.\d+)?"
    t = re.sub(rf"(?:₹|\bRs\.?|\bINR)\s?({money_num})", lambda m: f"{m.group(1)} {_w(lang, 'rupee')}", t)
    t = re.sub(rf"\$\s?({money_num})", lambda m: f"{m.group(1)} {_w(lang, 'dollar')}", t)
    t = re.sub(rf"({money_num})\s?%", lambda m: f"{m.group(1)} {_w(lang, 'percent')}", t)
    t = re.sub(rf"({_NUM})\.(\d+)", dec, t)
    t = re.sub(r"(\d)\s*&\s*(\d)", rf"\1 {_w(lang, 'and')} \2", t)
    t = re.sub(r"\s&\s", f" {_w(lang, 'and')} ", t)

    def plain(m):
        s = m.group()
        n = int(s.replace(",", ""))
        if lang == "hi" and "," not in s and len(s) == 4 and 1100 <= n <= 1999:
            return _year_hi(n)
        return num_words(n, lang)
    t = re.sub(_NUM, plain, t)
    return t


_NEMO = {}


MONTHS_EN = "January February March April May June July August September October November December".split()


# ddmmyyyy with no separators is read as a date only right after a date word, elsewhere it stays a number
_DATE_WORDS = {"en": r"dated?|dates|dob|d\.o\.b\.?|born|on|birthday|due|expir\w*|valid (?:till|until|upto)|from|till|until",
               "in": r"दिनांक|तारीख़?|तिथि|जन्मतिथि|जन्म तिथि|dob|date"}


def _date8(t: str, lang: str) -> str:
    """'DOB 04042004' -> 'DOB 04/04/2004' (the dd/mm/yyyy rule then reads it). Valid day, month, year 1900-2099 only."""
    def f(m):
        d, mo, y = int(m.group(3)), int(m.group(4)), int(m.group(5))
        try:
            ok = 1900 <= y <= 2099 and bool(datetime.date(y, mo, d))     # real calendar day (no 31 Feb)
        except ValueError:
            ok = False
        return f"{m.group(1)}{m.group(2)}{m.group(3)}/{m.group(4)}/{m.group(5)}" if ok else m.group()
    w = _DATE_WORDS["en" if lang == "en" else "in"]
    gap = r"(?:\W{0,3}\s*(?:is|was|of|है|था|:)?\W{0,3}\s*)"           # 'DOB is 04042004', 'दिनांक: 15082024'
    return re.sub(rf"(?i)\b({w})({gap})(\d{{2}})(\d{{2}})(\d{{4}})\b", f, t)


def _en_pre(t: str) -> str:
    """Fixes for what NeMo's English grammar gets wrong on Indian / web text: day-first dates, rupees,
    URLs and e-mails, # numbers, degrees, km/h, decades, version strings, minus signs, 'X-Y days' ranges."""
    def date(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{MONTHS_EN[mo - 1]} {d}, {y}" if 1 <= d <= 31 and 1 <= mo <= 12 else m.group()
    t = _date8(t, "en")
    t = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", date, t)          # India: day first
    t = re.sub(r"(https?://\S+|www\.\S+)", lambda m: _spell_net(m.group().rstrip(".,"), "en"), t)
    t = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", lambda m: _spell_net(m.group(), "en"), t)
    t = re.sub(r"\bwww\b", "double-u double-u double-u", t)   # "w w w" was read as two w
    t = re.sub(r"\+?\d[\d -]{8,}\d", lambda m: ("plus " if m.group().startswith("+") else "") + " ".join(c for c in m.group() if c.isdigit())
               if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), t)          # phone numbers digit by digit
    t = re.sub(rf"(?:₹|\bRs\.?|\bINR)\s?((?:{_NUM})(?:\.\d+)?)", r"\1 rupees", t)
    t = re.sub(r"#\s?(\d+)", r"number \1", t)
    t = re.sub(r"\s?°\s?C\b", " degrees Celsius", t)
    t = re.sub(r"\s?°\s?F\b", " degrees Fahrenheit", t)
    t = re.sub(r"\bkm/h\b|\bkmph\b", "kilometers per hour", t)
    t = re.sub(r"\bmph\b", "miles per hour", t)

    def decade(m):
        from num2words import num2words
        w = num2words(int(m.group(1)), lang="en", to="year").replace("-", " ").split()
        w[-1] = w[-1][:-1] + "ies" if w[-1].endswith("y") else w[-1] + "s"
        return " ".join(w)
    t = re.sub(r"\b(1[0-9]{2}0|20[0-9]0)s\b", decade, t)
    t = re.sub(r"\b\d+(?:\.\d+){2,}\b", lambda m: " point ".join(m.group().split(".")), t)
    t = re.sub(r"(?<![\w-])-(\d)", r"minus \1", t)
    t = re.sub(r"\b(\d+)\s?-\s?(\d+)(\s+(?:days|weeks|months|years|hours|minutes|seconds|times|people|pages))", r"\1 to \2\3", t)
    return t


def _en(text: str) -> str:
    text = _en_pre(text)
    try:
        if "n" not in _NEMO:
            import logging
            from nemo_text_processing.text_normalization.normalize import Normalizer
            logging.getLogger("NeMo-text-processing").setLevel(logging.ERROR)
            _NEMO["n"] = Normalizer(input_case="cased", lang="en")
        return "\n".join(_NEMO["n"].normalize(x, punct_post_process=True) if x.strip() else x for x in text.split("\n"))
    except ImportError:
        t = re.sub(r"(https?://\S+|www\.\S+)", lambda m: _spell_net(m.group().rstrip(".,"), "en"), text)
        t = re.sub(r"\$\s?(\d+)", r"\1 dollars", t)
        t = re.sub(r"(\d+)\s?%", r"\1 percent", t)
        return re.sub(r"\d+", lambda m: num_words(m.group(), "en"), t)


def normalize(text: str, lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"lang {lang!r}: one of {LANGS}")
    t = unicodedata.normalize("NFC", text)
    t = "".join(ch for ch in t if ch in "°" or unicodedata.category(ch) not in ("So", "Cs", "Co"))   # emoji, private use
    t = t.replace("​", "").replace("॥", "।")
    t = re.sub(r"[“”]", '"', re.sub(r"[‘’]", "'", t))
    t = _en(t) if lang == "en" else _indic(t, lang)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r" *\n *", "\n", t).strip()


_END = re.compile(r"(?<=[.!?।;])[\"')\]]*\s+|\n+")


def split(text: str, lang: str, max_chars: int | None = None, min_chars: int | None = None) -> list[str]:
    mx = max_chars or MAX_CHARS.get(lang, MAX_CHARS_IN)
    min_chars = min_chars or MIN_CHARS.get(lang, MIN_CHARS_IN)
    pieces: list[str] = []
    for s in (x.strip() for x in _END.split(text)):
        if not s:
            continue
        if len(s) <= mx:
            pieces.append(s)
            continue
        cur = ""
        for c in re.split(r"(?<=[,:،])\s+", s):                # long sentence: commas first, then words
            words = c.split() if len(c) > mx else [c]
            for w in words:
                if cur and len(cur) + 1 + len(w) > mx:
                    pieces.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
        if cur:
            pieces.append(cur)
    out: list[str] = []
    for p in pieces:                                             # merge short pieces forward up to the limit
        if out and (len(out[-1]) < min_chars or len(p) < min_chars // 3) and len(out[-1]) + 1 + len(p) <= mx:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out
