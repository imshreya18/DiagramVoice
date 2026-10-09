import os, json, io, re, random, textwrap
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from gtts import gTTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-flash-latest"]

EXTRACT_PROMPT = """You are describing a diagram for a blind student.
Return ONLY valid JSON with keys: type ("graph"/"circuit"/"geometry"/"biology"/"physics"/"other"), title,
x_axis {label, unit, range}, y_axis {label, unit, range},
curves [{shape, slope, starts_at, ends_at, points}],
components [circuit parts and connections, else []],
parts [{label, description, position}] for geometry, biology and physics diagrams (else []). Include EVERY printed label,
  including small labels and sub-labels written under or inside a group label (for example a group label
  "Root system" with "Taproot" and "Lateral roots" written under it gives three parts),
legend [for a printed legend or key, usually a box in a corner: one entry per item, giving its line style or colour
  and what it stands for, else []],
relationships [how parts connect, flow, or relate, else []],
measurements [lengths, angles, formulas or values printed on a geometry or physics diagram, else []],
key_features [short plain sentences in English], uncertain [things not clearly readable],
plot {kind: "bar"/"line"/"none", categories: [bar labels], x_values: [numbers], y_values: [numbers], schematic: true/false}.
RULES: List every printed label. Do not skip small, crowded or sub-labels. If a legend or key exists, read all its entries,
and make one of the key_features a sentence that lists them. Never invent numbers. If a value is not printed, put it in "uncertain".
Only name parts that have a printed label. For an unlabelled part say "an unlabelled part" and put it in "uncertain".
PLOT RULES (used to redraw the diagram for tactile printing):
- bar chart: kind "bar", categories and y_values taken from the printed values, schematic false.
- line graph with visible numbers: kind "line", real x_values and y_values, schematic false.
- line graph with NO numbers: kind "line", 2 to 6 points on a 0 to 1 scale that show only the shape, schematic true.
- circuits or anything else: kind "none"."""

GOOD = {"model": None}


def clean_math(t):
    """Turn LaTeX-style leftovers like \\(F_{n}\\) into plain speakable text."""
    t = str(t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$", "", t)
    t = re.sub(r"\\(?:mathrm|text|mathbf)\{([^}]*)\}", r"\1", t)
    for a, b in (("\\theta", "theta"), ("\\mu", "mu"), ("\\cdot", " times "),
                 ("\\times", " times "), ("\\sin", "sin"), ("\\cos", "cos"),
                 ("\\tan", "tan")):
        t = t.replace(a, b)
    t = re.sub(r"\^\{?2\}?", " squared", t)
    t = re.sub(r"\^\{?3\}?", " cubed", t)
    t = re.sub(r"\^\{?(-?\d+)\}?", r" to the power \1", t)
    t = re.sub(r"_\{?([A-Za-z0-9]+)\}?", r" \1", t)
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    t = re.sub(r"\(\(([^()]*)\)\)", r"(\1)", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def call_llm(contents, as_json=False, temperature=None):
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if as_json else None,
        temperature=temperature,
        http_options=types.HttpOptions(timeout=15000),
    )
    order = MODELS if GOOD["model"] is None else \
        [GOOD["model"]] + [m for m in MODELS if m != GOOD["model"]]
    for model in order:
        try:
            text = client.models.generate_content(
                model=model, contents=contents, config=cfg).text
            GOOD["model"] = model
            return text if as_json else clean_math(text)
        except Exception:
            continue
    raise RuntimeError("The AI service is busy. Please try again in a minute.")


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def shrink(img, max_side=1024):
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def make_description(data, lang):
    prompt = f"""Write a spoken description of this diagram for a blind student,
in {lang} (written in that language's own script). Use this fixed order: what kind of diagram it is,
its overall shape, key labelled parts, key values, what it means.
Only if the uncertain list is not empty, mention those items honestly.
If it is empty, do not say anything about uncertainty.
Say "approximately" for estimated values.Name every labelled part and,
if the data has a legend, say what it shows, do not miss any lebel. Plain sentences only, no markdown, under 190 words.
Diagram data: {json.dumps(data)}"""
    return call_llm([prompt])


def get_pack(data, lang):
    """Description, key facts and uncertain items in the chosen language.
    Built once from the English extraction, then cached per language.key facts should 3 to 6 points"""
    packs = st.session_state.setdefault("packs", {})
    if lang in packs:
        return packs[lang]
    en_desc = clean_math(data.get("spoken_description") or make_description(data, "English"))
    en_keys = [clean_math(f) for f in data.get("key_features", [])]
    en_unsure = [clean_math(u) for u in data.get("uncertain", [])]
    if lang == "English":
        pack = {"description": en_desc, "key_features": en_keys, "uncertain": en_unsure}
    else:
        prompt = f"""Translate the following into {lang}, written in that language's own script.
Keep the meaning exact. Keep numbers, units and the names of labelled parts accurate.
Plain sentences only: no markdown and no LaTeX or math markup.
Return ONLY JSON: {{"description": str, "key_features": [str], "uncertain": [str]}}
Input: {json.dumps({"description": en_desc, "key_features": en_keys, "uncertain": en_unsure})}"""
        out = parse_json(call_llm([prompt], as_json=True))
        pack = {"description": clean_math(out.get("description") or en_desc),
                "key_features": [clean_math(x) for x in (out.get("key_features") or en_keys)],
                "uncertain": [clean_math(x) for x in (out.get("uncertain") or [])]}
    packs[lang] = pack
    return pack


LANGS = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Bengali": "bn", "Tamil": "ta",
         "Telugu": "te", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml"}


@st.cache_data(show_spinner=False)
def make_audio(text, lang, tld="com", slow=False):
    buf = io.BytesIO()
    gTTS(text=text, lang=LANGS.get(lang, "en"),
         tld=tld if lang == "English" else "com", slow=slow).write_to_fp(buf)
    return buf.getvalue()


ACCENTS = {"Indian": "co.in", "American": "com", "British": "co.uk", "Australian": "com.au"}


def speak(text, lang):
    """Speech using the voice settings the user picked on the page."""
    tld = ACCENTS.get(st.session_state.get("accent", "Indian"), "co.in")
    slow = st.session_state.get("speed", "Normal") == "Slow"
    return make_audio(text, lang, tld, slow)


def transcribe(audio_bytes, lang):
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    return call_llm([part, f"Transcribe this spoken question exactly. "
                           f"The language is probably {lang}. Return only the text."]).strip()


# ---------- Tactile-ready export ----------
def make_tactile(data):
    """Redraw the diagram as a simple, thick-line, high-contrast figure.
    Returns (png_bytes, svg_bytes, pdf_bytes) or None if not possible."""
    plot = data.get("plot") or {}
    kind = plot.get("kind")
    try:
        if kind == "bar":
            cats = [str(c) for c in plot.get("categories", [])]
            vals = [float(v) for v in plot.get("y_values", [])]
            if not cats or len(cats) != len(vals):
                return None
        elif kind == "line":
            xs = [float(v) for v in plot.get("x_values", [])]
            ys = [float(v) for v in plot.get("y_values", [])]
            if len(xs) < 2 or len(xs) != len(ys):
                return None
        else:
            return None
    except (TypeError, ValueError):
        return None

    schematic = bool(plot.get("schematic"))
    xlabel = (data.get("x_axis") or {}).get("label") or ""
    ylabel = (data.get("y_axis") or {}).get("label") or ""
    title = data.get("title") or ""

    fig, ax = plt.subplots(figsize=(8, 6))
    if kind == "bar":
        hatches = ["//", "\\\\", "xx", "..", "++", "oo"]
        bars = ax.bar(cats, vals, facecolor="white", edgecolor="black", linewidth=4,
                      hatch=None)
        for i, b in enumerate(bars):
            b.set_hatch(hatches[i % len(hatches)])
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{vals[i]:g}", ha="center", va="bottom",
                    fontsize=22, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
    else:
        ax.plot(xs, ys, color="black", linewidth=7,
                marker=None if schematic else "s", markersize=16)
        if schematic:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, -0.18, "Schematic: shape only, no exact values",
                    transform=ax.transAxes, ha="center", fontsize=16)

    ax.set_xlabel(xlabel, fontsize=24, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=26, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=20, width=4, length=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(6)
    fig.tight_layout()

    out = []
    for fmt in ("png", "svg", "pdf"):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=150, facecolor="white")
        out.append(buf.getvalue())
    plt.close(fig)
    return tuple(out)


# ---------- Braille (English, Grade 1 / uncontracted) ----------
_BR = dict(zip("abcdefghijklmnopqrstuvwxyz", "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵"))
_BRD = dict(zip("1234567890", "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚"))
_BRP = {".": "⠲", ",": "⠂", ";": "⠆", ":": "⠒", "!": "⠖", "?": "⠦", "'": "⠄",
        "-": "⠤", "(": "⠐⠣", ")": "⠐⠜", "/": "⠸⠌", "=": "⠐⠶", "+": "⠐⠖"}
_BR_WORDS = {"°": " degrees", "Ω": " ohms", "Ω": " ohms", "%": " percent", "×": " times ",
             "÷": " divided by ", "µ": "micro", "μ": "micro", "±": " plus or minus ",
             "≈": " approximately ", "≤": " less than or equal to ", "≥": " greater than or equal to ",
             "<": " less than ", ">": " greater than ", "→": " to ", "−": "-", "–": "-", "—": " - ",
             "’": "'", "‘": "'", "“": '"', "”": '"', "√": " square root of ", "π": " pi ",
             "θ": " theta ", "α": " alpha ", "β": " beta ", "λ": " lambda ", "Δ": " delta ",
             "²": " squared", "³": " cubed", "&": " and ", "*": " times ", "_": " "}


def to_braille(text):
    text = str(text).replace('"', "")
    for a, b in _BR_WORDS.items():
        text = text.replace(a, b)
    text = re.sub(r"[ \t]+", " ", text)
    out, in_num, i, n = [], False, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _BRD:
            if not in_num:
                out.append("⠼")
                in_num = True
            out.append(_BRD[ch])
            i += 1
            continue
        if in_num and ch in ".," and i + 1 < n and text[i + 1] in _BRD:
            out.append(_BRP[ch])
            i += 1
            continue
        was_num, in_num = in_num, False
        low = ch.lower()
        if low in _BR:
            if was_num and low in "abcdefghij":
                out.append("⠰")
            if ch.isupper():
                if i + 1 < n and text[i + 1].isupper() and text[i + 1].lower() in _BR:
                    j = i
                    while j < n and text[j].isupper() and text[j].lower() in _BR:
                        j += 1
                    out.append("⠠⠠" + "".join(_BR[c.lower()] for c in text[i:j]))
                    i = j
                    continue
                out.append("⠠")
            out.append(_BR[low])
        elif ch in _BRP:
            out.append(_BRP[ch])
        elif ch == "\n":
            out.append("\n")
        elif ch.isspace():
            out.append("⠀")
        i += 1
    return "".join(out)


def braille_summary(data):
    lines = [str(data.get("title") or "")]
    lines += [str(f) for f in data.get("key_features", [])]
    wrapped = [textwrap.fill(l, 36) for l in lines if l.strip()]
    return to_braille("\n\n".join(wrapped))


# ---------- Quiz ----------
QUIZ_FOCUS = [
    "reading exact values from the diagram",
    "comparing two parts of the diagram (higher, lower, difference)",
    "what the diagram means in a real-life situation",
    "definitions and the underlying concept",
    "simple calculations using the values in the diagram",
    "common mistakes students make with this kind of diagram",
    "what would change if a value changed",
]


def make_quiz(data, lang, avoid=None):
    focus = "; ".join(random.sample(QUIZ_FOCUS, 3))
    avoid_txt = ""
    if avoid:
        avoid_txt = ("\nDo NOT repeat or rephrase any of these earlier questions:\n- " +
                     "\n- ".join(avoid[-20:]))
    prompt = f"""Create 5 NEW multiple-choice questions for a school student, in {lang}
(written in that language's own script), based on this diagram data:
{json.dumps(data)}
Emphasise these angles this time: {focus}.{avoid_txt}
Questions about the diagram must use only values present in the data, never invent numbers.
Include at least 1 concept question. Each question has exactly 4 options.
Put the correct answer at a random position (not always the same index).
Return ONLY a JSON list. Each item: {{"question": str, "options": [4 strings],
"answer_index": 0-3, "explanation": one short sentence}}.
Plain text only: no markdown and no LaTeX or math markup. Write symbols as words
(for example write "normal force" or "F n", never "F n with LaTeX")."""
# (for example write "normal force" or "F n", never "\\(F_{n}\\)")."""
    result = parse_json(call_llm([prompt], as_json=True, temperature=1.0))
    if isinstance(result, dict):
        result = result.get("questions", [])
    good = []
    for q in result:
        try:
            if len(q["options"]) == 4 and 0 <= int(q["answer_index"]) <= 3:
                q["answer_index"] = int(q["answer_index"])
                q["question"] = clean_math(q["question"])
                q["options"] = [clean_math(o) for o in q["options"]]
                q["explanation"] = clean_math(q.get("explanation", ""))
                good.append(q)
        except (KeyError, TypeError, ValueError):
            continue
    return good


st.set_page_config(page_title="DiagramVoice", page_icon="🔊", layout="wide")

### FONT ##

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Poppins:wght@600;700;800&family=Noto+Sans+Devanagari:wght@400;700&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button {
    font-family: 'Atkinson Hyperlegible', 'Noto Sans Devanagari, pop;
}
html, body { font-size: 20px; }
h1, h2, h3, h4, .hero-title {
    font-family: 'Poppins', 'Noto Sans Devanagari', pop!important;
    font-weight: 700;
}
footer { visibility: hidden; }
.block-container { padding-top: 4.5rem; max-width: 1250px; }

/* soft colour wash at the top that works in light and dark mode */
.stApp::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 380px;
    background: linear-gradient(180deg, rgba(99,102,241,.16), rgba(99,102,241,0));
    pointer-events: none; z-index: 0; }

/* ---------- animations ---------- */
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes shift  { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes bounce { 0%, 100% { height: 10px; } 50% { height: 38px; } }
@keyframes glow   { 0%, 100% { box-shadow: 0 0 0 0 rgba(6,182,212,.5); } 50% { box-shadow: 0 0 0 10px rgba(6,182,212,0); } }

.hero { background: linear-gradient(120deg, #312e81, #2563eb, #06b6d4, #2563eb, #312e81);
        background-size: 300% 300%; animation: shift 14s ease infinite, fadeUp .7s ease both;
        padding: 2rem 2.2rem; border-radius: 24px; margin-bottom: 1.4rem;
        box-shadow: 0 14px 34px rgba(37, 99, 235, .28); position: relative; overflow: hidden; }
.hero-title { color: #ffffff !important; font-size: 2.7rem; font-weight: 800; line-height: 1.15; }
.hero-sub { color: #e0f2fe; font-size: 1.3rem; margin-top: .5rem; max-width: 760px; }
.badges span { display: inline-block; background: rgba(255,255,255,.2); color: #fff;
        padding: .35rem .9rem; border-radius: 999px; margin: .9rem .45rem 0 0; font-size: .95rem;
        transition: transform .2s ease, background .2s ease; }
.badges span:hover { transform: translateY(-3px); background: rgba(255,255,255,.32); }
.eq { position: absolute; right: 2rem; top: 1.6rem; display: flex; gap: 6px; align-items: flex-end; height: 40px; }
.eq i { display: block; width: 7px; height: 10px; border-radius: 4px; background: rgba(255,255,255,.85);
        animation: bounce 1.1s ease-in-out infinite; }
.eq i:nth-child(2) { animation-delay: .15s; } .eq i:nth-child(3) { animation-delay: .3s; }
.eq i:nth-child(4) { animation-delay: .45s; } .eq i:nth-child(5) { animation-delay: .6s; }
@media (max-width: 700px) { .eq { display: none; } .hero-title { font-size: 2rem; } }

[data-testid="stVerticalBlockBorderWrapper"] { animation: fadeUp .6s ease both; }

.step { background: rgba(99,102,241,.12); border-left: 6px solid #2563eb; border-radius: 14px;
        padding: 1rem 1.2rem; transition: transform .2s ease, box-shadow .2s ease; }
.step:hover { transform: translateX(6px); box-shadow: 0 8px 20px rgba(37,99,235,.15); }
.step b { font-family: 'Poppins', Courier New; font-size: 1.1rem; }

.stButton > button, .stDownloadButton > button {
        font-size: 1.1rem; font-weight: 700; padding: .7rem 1.4rem; border-radius: 14px;
        transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 18px rgba(37,99,235,.25); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg, #2563eb, #06b6d4); border: none; color: #fff;
        animation: glow 2.4s ease-in-out infinite; }
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible {
        outline: 4px solid #f59e0b !important; outline-offset: 2px; }

/* respect users who turn animations off */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
<div class="hero">
  <div class="eq"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="hero-title"> DiagramVoice</div>
  <div class="hero-sub">Hear any textbook graph or circuit. Ask your doubts by voice, in Hindi or English.</div>
  <div class="badges"><span>Hindi + English</span><span>Voice in, voice out</span>
  <span>Never invents numbers</span><span>Tactile-ready export</span><span>Free to use</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.35], gap="large")

# ======================= LEFT: upload and settings =======================

with left, st.container(border=True):
    st.subheader("1. Choose your diagram")
    lang = st.selectbox("Language", list(LANGS))
    vc1, vc2 = st.columns(2)
    vc1.selectbox("Voice accent (English only)", list(ACCENTS), key="accent")
    vc2.radio("Speaking speed", ["Normal", "Slow"], key="speed", horizontal=True)
    file = st.file_uploader("Upload a photo, image, or PDF of the diagram",
                            type=["png", "jpg", "jpeg", "pdf"])

    sig = (file.name, file.size) if file else None
    if st.session_state.get("file_sig") != sig:
        for k in list(st.session_state.keys()):
            if k in ("data", "packs", "tactile", "qa_cache", "quiz", "quiz_history",
                     "voice_id", "voice_q") or str(k).startswith("quiz_"):
                st.session_state.pop(k, None)
        st.session_state["file_sig"] = sig

    img = None
    if file:
        if file.name.lower().endswith(".pdf"):
            import fitz  # PyMuPDF
            pdf = fitz.open(stream=file.getvalue(), filetype="pdf")
            page_no = 1
            if len(pdf) > 1:
                page_no = st.number_input(
                    f"This PDF has {len(pdf)} pages. Which page has the diagram?",
                    min_value=1, max_value=len(pdf), value=1, step=1)
            pix = pdf[int(page_no) - 1].get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
        else:
            img = Image.open(file)

    if img is not None:
        st.image(img, caption="Your diagram")
        if st.button("🔊 Describe this diagram", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading the diagram..."):
                    extra = f"""
Also add a key "spoken_description": a spoken description in English,
for a blind student, under 120 words, plain sentences, no markdown, in this order:
kind of diagram, overall shape or layout, key labelled parts and where they are, how parts connect or relate, key values, what it means.
Only mention uncertain items if the uncertain list is not empty; say "approximately" for estimated values."""
                    data = parse_json(call_llm([shrink(img), EXTRACT_PROMPT + extra], as_json=True))
                    tactile = make_tactile(data)
                st.session_state.update(data=data, tactile=tactile)
                st.session_state.pop("packs", None)
                st.session_state.pop("qa_cache", None)
                st.session_state.pop("quiz", None)
                st.session_state.pop("quiz_history", None)
                for k in [k for k in st.session_state if str(k).startswith("quiz_")
                          and k != "quiz_history"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Something went wrong: {e}")


# ======================= RIGHT: results =======================
with right, st.container(border=True):
    if "data" not in st.session_state:
        st.subheader("How it works")
        st.markdown("""
<div class="step"><b>1. Upload</b><br>A photo, image, or PDF page of a graph, bar chart, or circuit.</div><br>
<div class="step"><b>2. Listen</b><br>Get a clear spoken description in the same order every time.</div><br>
<div class="step"><b>3. Ask and practise</b><br>Ask doubts by voice, download a tactile-ready version, and take a quiz.</div>
""", unsafe_allow_html=True)
    else:
        st.subheader("2. Your results")
        view = st.radio("Section",
                        ["🔊 Listen", "✋ Tactile", "🎤 Ask", "📝 Quiz", "🔍 Under the hood"],
                        horizontal=True, key="view", label_visibility="collapsed")
        slang = lang  # everything follows the language selected on the left

        if view == "🔊 Listen":
            try:
                with st.spinner(f"Preparing {lang}..."):
                    pack = get_pack(st.session_state["data"], lang)
                st.write(pack["description"])
                st.audio(speak(pack["description"], lang), format="audio/mp3")
                if pack["uncertain"]:
                    st.warning("Not sure about: " + "; ".join(pack["uncertain"]))
                st.markdown("**Key facts**")
                for f in pack["key_features"]:
                    st.write("• " + f)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

        elif view == "✋ Tactile":
            tac = st.session_state.get("tactile")
            if tac:
                png, svg, pdf_bytes = tac
                st.image(png, caption="Simplified thick-line version for swell paper "
                                      "or for a school to emboss")
                c1, c2 = st.columns(2)
                c1.download_button("⬇ Download SVG", svg, file_name="tactile_diagram.svg",
                                   mime="image/svg+xml", use_container_width=True)
                c2.download_button("⬇ Download PDF", pdf_bytes, file_name="tactile_diagram.pdf",
                                   mime="application/pdf", use_container_width=True)
                st.caption("A simplified redraw from the extracted data, ready for tactile "
                           "printing. It is not a tactile device.")
            else:
                st.info("A tactile drawing is available for graphs and bar charts. "
                        "It is not available for this diagram type yet.")

            st.markdown("**Braille-ready text** (English, Grade 1 uncontracted)")
            braille = braille_summary(st.session_state["data"])
            if braille.strip():
                st.text(braille)
                st.download_button("⬇ Download Braille text (.txt)", braille.encode("utf-8"),
                                   file_name="braille_summary.txt", mime="text/plain",
                                   use_container_width=True)
                st.caption("Unicode Braille of the title and key facts. Not yet checked by a "
                           "certified Braille transcriber. Embosser software may need a .brf conversion.")

        elif view == "🎤 Ask":
            voice = st.audio_input("🎤 Ask by voice")
            typed = st.text_input("Or type your question")
            question = typed
            try:
                if voice is not None:
                    audio_bytes = voice.getvalue()
                    vid = hash(audio_bytes)
                    if st.session_state.get("voice_id") != vid:
                        with st.spinner("Listening..."):
                            st.session_state["voice_q"] = transcribe(audio_bytes, slang)
                        st.session_state["voice_id"] = vid
                    if not typed:
                        question = st.session_state["voice_q"]

                if question:
                    cache = st.session_state.setdefault("qa_cache", {})
                    key = (question, slang)
                    if key not in cache:
                        with st.spinner("Thinking..."):
                            ans = call_llm([f"""You are a patient school tutor helping a blind student.
Answer in {slang}, written in that language's own script.

The student is looking at this diagram (extracted data):
{json.dumps(st.session_state['data'])}

RULES:
1. If the question is about THIS diagram (its values, labels, shape, which is highest, etc.),
   answer ONLY from the diagram data. If the value is not in the data, say you are not sure.
   Never invent numbers.
2. If the question is a general concept or doubt (for example "what is a bar chart",
   "what is acceleration", "how do I find slope"), explain it from your own knowledge
   in simple words, at school level, with one short everyday example.
   Where useful, connect it to this diagram.
3. If the question is far outside school subjects, politely say you can only help with
   study doubts related to this topic.
4. Use the same names for parts as the labels in the diagram data (for example "normal force",
   not "vertical force"). No LaTeX or math markup.
5. Speak in plain sentences only. No markdown, bullet points, or symbols.
   Keep it under 100 words, because this will be read aloud.

Question: {question}"""])
                            cache[key] = (ans, None)
                    ans, _ = cache[key]
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {ans}")
                    st.audio(speak(ans, slang), format="audio/mp3")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

        elif view == "📝 Quiz":
            if st.button("Create a quiz on this diagram", type="primary"):
                try:
                    with st.spinner("Writing questions..."):
                        history = st.session_state.get("quiz_history", [])
                        quiz = make_quiz(st.session_state["data"], slang, avoid=history)
                        st.session_state["quiz_history"] = history + [q["question"] for q in quiz]
                    for k in [k for k in st.session_state if str(k).startswith("quiz_")
                              and k not in ("quiz_history",)]:
                        st.session_state.pop(k, None)
                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_lang"] = slang
                    if not quiz:
                        st.warning("Could not make a quiz this time. Please try again.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

            quiz = st.session_state.get("quiz")
            if quiz and st.session_state.get("quiz_lang") != slang:
                st.info(f"This quiz was made in {st.session_state.get('quiz_lang')}. "
                        f"Click “Create a quiz” to get questions in {slang}.")
                quiz = None
            if quiz:
                if st.checkbox("🔊 Read the quiz aloud"):
                    spoken = " ".join(
                        f"Question {i + 1}. {q['question']}. Options: " +
                        ". ".join(f"{'ABCD'[j]}, {o}" for j, o in enumerate(q["options"]))
                        for i, q in enumerate(quiz))
                    st.audio(speak(spoken, slang), format="audio/mp3")

                for i, q in enumerate(quiz):
                    st.radio(f"Question {i + 1}. {q['question']}", q["options"],
                             index=None, key=f"quiz_{i}")

                if st.button("Check my answers"):
                    score = 0
                    for i, q in enumerate(quiz):
                        chosen = st.session_state.get(f"quiz_{i}")
                        correct = q["options"][q["answer_index"]]
                        if chosen == correct:
                            score += 1
                            st.success(f"Question {i + 1}: correct. {q.get('explanation', '')}")
                        else:
                            st.error(f"Question {i + 1}: the correct answer is {correct}. "
                                     f"{q.get('explanation', '')}")
                    st.subheader(f"Your score: {score} out of {len(quiz)}")

        else:
            st.caption("The AI first extracts structured data from the diagram, then every "
                       "explanation is written from this data. This is why it does not invent numbers.")
            st.json(st.session_state["data"])

st.divider()
st.caption("AI can make mistakes. Please check important answers with your teacher. "
           "DiagramVoice · HackNova 2026 · Inclusive Technology")


