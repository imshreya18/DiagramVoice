import os, json, io, re, random
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
Return ONLY valid JSON with keys: type ("graph"/"circuit"/"other"), title,
x_axis {label, unit, range}, y_axis {label, unit, range},
curves [{shape, slope, starts_at, ends_at, points}],
components [circuit parts and connections, else []],
key_features [short plain sentences], uncertain [things not clearly readable],
plot {kind: "bar"/"line"/"none", categories: [bar labels], x_values: [numbers], y_values: [numbers], schematic: true/false}.
RULES: Never invent numbers. If a value is not printed, put it in "uncertain".
PLOT RULES (used to redraw the diagram for tactile printing):
- bar chart: kind "bar", categories and y_values taken from the printed values, schematic false.
- line graph with visible numbers: kind "line", real x_values and y_values, schematic false.
- line graph with NO numbers: kind "line", 2 to 6 points on a 0 to 1 scale that show only the shape, schematic true.
- circuits or anything else: kind "none"."""

GOOD = {"model": None}


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
            return text
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
in {lang} (Devanagari script if Hindi). Use this fixed order: what kind of diagram it is,
its overall shape, key labelled parts, key values, what it means.
Only if the uncertain list is not empty, mention those items honestly.
If it is empty, do not say anything about uncertainty.
Say "approximately" for estimated values. Plain sentences only, no markdown, under 120 words.
Diagram data: {json.dumps(data)}"""
    return call_llm([prompt])


@st.cache_data(show_spinner=False)
def make_audio(text, lang, tld="com", slow=False):
    buf = io.BytesIO()
    gTTS(text=text, lang="hi" if lang == "Hindi" else "en",
         tld=tld, slow=slow).write_to_fp(buf)
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
(Devanagari script if Hindi), based on this diagram data:
{json.dumps(data)}
Emphasise these angles this time: {focus}.{avoid_txt}
Questions about the diagram must use only values present in the data, never invent numbers.
Include at least 1 concept question. Each question has exactly 4 options.
Put the correct answer at a random position (not always the same index).
Return ONLY a JSON list. Each item: {{"question": str, "options": [4 strings],
"answer_index": 0-3, "explanation": one short sentence}}.
Plain text only, no markdown."""
    result = parse_json(call_llm([prompt], as_json=True, temperature=1.0))
    if isinstance(result, dict):
        result = result.get("questions", [])
    good = []
    for q in result:
        try:
            if len(q["options"]) == 4 and 0 <= int(q["answer_index"]) <= 3:
                q["answer_index"] = int(q["answer_index"])
                good.append(q)
        except (KeyError, TypeError, ValueError):
            continue
    return good


st.set_page_config(page_title="DiagramVoice", page_icon="🔊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Poppins:wght@600;700;800&family=Noto+Sans+Devanagari:wght@400;700&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button {
    font-family: 'Atkinson Hyperlegible', 'Noto Sans Devanagari', sans-serif;
}
html, body { font-size: 18px; }
h1, h2, h3, h4, .hero-title {
    font-family: 'Poppins', 'Noto Sans Devanagari', sans-serif !important;
    font-weight: 700; color: #1e1b4b;
}
.stApp { background: linear-gradient(180deg, #eef2ff 0%, #ffffff 380px); }
footer { visibility: hidden; }
.block-container { padding-top: 4.5rem; max-width: 1250px; }

.hero { background: linear-gradient(120deg, #312e81 0%, #2563eb 55%, #06b6d4 100%);
        padding: 2rem 2.2rem; border-radius: 24px; margin-bottom: 1.4rem;
        box-shadow: 0 14px 34px rgba(37, 99, 235, .28); }
.hero-title { color: #ffffff !important; font-size: 2.7rem; font-weight: 800; line-height: 1.15; }
.hero-sub { color: #e0f2fe; font-size: 1.3rem; margin-top: .5rem; }
.badges span { display: inline-block; background: rgba(255,255,255,.2); color: #fff;
        padding: .35rem .9rem; border-radius: 999px; margin: .9rem .45rem 0 0; font-size: .95rem; }

.step { background: #eef2ff; border-left: 6px solid #2563eb; border-radius: 14px;
        padding: 1rem 1.2rem; color: #0f172a; }
.step b { font-family: 'Poppins', sans-serif; font-size: 1.1rem; color: #1e1b4b; }

.stButton > button, .stDownloadButton > button {
        font-size: 1.1rem; font-weight: 700; padding: .7rem 1.4rem; border-radius: 14px; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg, #2563eb, #06b6d4); border: none; color: #fff; }
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible {
        outline: 4px solid #f59e0b !important; outline-offset: 2px; }
</style>
<div class="hero">
  <div class="hero-title">🔊 DiagramVoice</div>
  <div class="hero-sub">Hear any textbook graph or circuit. Ask your doubts by voice, in Hindi or English.</div>
  <div class="badges"><span>Hindi + English</span><span>Voice in, voice out</span>
  <span>Never invents numbers</span><span>Tactile-ready export</span><span>Free to use</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.35], gap="large")

# ======================= LEFT: upload and settings =======================
with left, st.container(border=True):
    st.subheader("1. Choose your diagram")
    lang = st.radio("Language", ["English", "Hindi"], horizontal=True)
    vc1, vc2 = st.columns(2)
    vc1.selectbox("Voice accent (English only)", list(ACCENTS), key="accent")
    vc2.radio("Speaking speed", ["Normal", "Slow"], key="speed", horizontal=True)
    file = st.file_uploader("Upload a photo, image, or PDF of the diagram",
                            type=["png", "jpg", "jpeg", "pdf"])

    sig = (file.name, file.size) if file else None
    if st.session_state.get("file_sig") != sig:
        for k in list(st.session_state.keys()):
            if k in ("data", "text", "lang", "tactile", "qa_cache", "quiz", "quiz_history",
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
Also add a key "spoken_description": a spoken description in {lang} (Devanagari if Hindi),
for a blind student, under 120 words, plain sentences, no markdown, in this order:
kind of diagram, overall shape, key labelled parts, key values, what it means.
Only mention uncertain items if the uncertain list is not empty; say "approximately" for estimated values."""
                    data = parse_json(call_llm([shrink(img), EXTRACT_PROMPT + extra], as_json=True))
                    text = data.get("spoken_description") or make_description(data, lang)
                    tactile = make_tactile(data)
                st.session_state.update(data=data, text=text, lang=lang, tactile=tactile)
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
    if "text" not in st.session_state:
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
        slang = st.session_state["lang"]

        if view == "🔊 Listen":
            st.write(st.session_state["text"])
            st.audio(speak(st.session_state["text"], slang), format="audio/mp3")
            if st.session_state["data"].get("uncertain"):
                st.warning("Not sure about: " +
                           "; ".join(map(str, st.session_state["data"]["uncertain"])))
            st.markdown("**Key facts**")
            for f in st.session_state["data"].get("key_features", []):
                st.write("• " + str(f))

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
                st.info("A tactile version is available for graphs and bar charts. "
                        "It is not available for this diagram type yet.")

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
Answer in {slang}. If the language is Hindi, write in Devanagari script.

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
4. Speak in plain sentences only. No markdown, bullet points, or symbols.
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
                    if not quiz:
                        st.warning("Could not make a quiz this time. Please try again.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

            quiz = st.session_state.get("quiz")
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


# import os, json, io, re, random
# import streamlit as st
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# from PIL import Image
# from gtts import gTTS
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# load_dotenv()
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
#           "gemini-3.5-flash", "gemini-flash-latest"]

# EXTRACT_PROMPT = """You are describing a diagram for a blind student.
# Return ONLY valid JSON with keys: type ("graph"/"circuit"/"other"), title,
# x_axis {label, unit, range}, y_axis {label, unit, range},
# curves [{shape, slope, starts_at, ends_at, points}],
# components [circuit parts and connections, else []],
# key_features [short plain sentences], uncertain [things not clearly readable],
# plot {kind: "bar"/"line"/"none", categories: [bar labels], x_values: [numbers], y_values: [numbers], schematic: true/false}.
# RULES: Never invent numbers. If a value is not printed, put it in "uncertain".
# PLOT RULES (used to redraw the diagram for tactile printing):
# - bar chart: kind "bar", categories and y_values taken from the printed values, schematic false.
# - line graph with visible numbers: kind "line", real x_values and y_values, schematic false.
# - line graph with NO numbers: kind "line", 2 to 6 points on a 0 to 1 scale that show only the shape, schematic true.
# - circuits or anything else: kind "none"."""

# GOOD = {"model": None}


# def call_llm(contents, as_json=False, temperature=None):
#     cfg = types.GenerateContentConfig(
#         response_mime_type="application/json" if as_json else None,
#         temperature=temperature,
#         http_options=types.HttpOptions(timeout=15000),
#     )
#     order = MODELS if GOOD["model"] is None else \
#         [GOOD["model"]] + [m for m in MODELS if m != GOOD["model"]]
#     for model in order:
#         try:
#             text = client.models.generate_content(
#                 model=model, contents=contents, config=cfg).text
#             GOOD["model"] = model
#             return text
#         except Exception:
#             continue
#     raise RuntimeError("The AI service is busy. Please try again in a minute.")


# def parse_json(text):
#     text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
#     return json.loads(text)


# def shrink(img, max_side=1024):
#     img = img.convert("RGB")
#     img.thumbnail((max_side, max_side))
#     return img


# def make_description(data, lang):
#     prompt = f"""Write a spoken description of this diagram for a blind student,
# in {lang} (Devanagari script if Hindi). Use this fixed order: what kind of diagram it is,
# its overall shape, key labelled parts, key values, what it means.
# Only if the uncertain list is not empty, mention those items honestly.
# If it is empty, do not say anything about uncertainty.
# Say "approximately" for estimated values. Plain sentences only, no markdown, under 120 words.
# Diagram data: {json.dumps(data)}"""
#     return call_llm([prompt])


# @st.cache_data(show_spinner=False)
# def make_audio(text, lang, tld="com", slow=False):
#     buf = io.BytesIO()
#     gTTS(text=text, lang="hi" if lang == "Hindi" else "en",
#          tld=tld, slow=slow).write_to_fp(buf)
#     return buf.getvalue()


# ACCENTS = {"Indian": "co.in", "American": "com", "British": "co.uk", "Australian": "com.au"}


# def speak(text, lang):
#     """Speech using the voice settings the user picked on the page."""
#     tld = ACCENTS.get(st.session_state.get("accent", "Indian"), "co.in")
#     slow = st.session_state.get("speed", "Normal") == "Slow"
#     return make_audio(text, lang, tld, slow)


# def transcribe(audio_bytes, lang):
#     part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
#     return call_llm([part, f"Transcribe this spoken question exactly. "
#                            f"The language is probably {lang}. Return only the text."]).strip()


# # ---------- Tactile-ready export ----------
# def make_tactile(data):
#     """Redraw the diagram as a simple, thick-line, high-contrast figure.
#     Returns (png_bytes, svg_bytes, pdf_bytes) or None if not possible."""
#     plot = data.get("plot") or {}
#     kind = plot.get("kind")
#     try:
#         if kind == "bar":
#             cats = [str(c) for c in plot.get("categories", [])]
#             vals = [float(v) for v in plot.get("y_values", [])]
#             if not cats or len(cats) != len(vals):
#                 return None
#         elif kind == "line":
#             xs = [float(v) for v in plot.get("x_values", [])]
#             ys = [float(v) for v in plot.get("y_values", [])]
#             if len(xs) < 2 or len(xs) != len(ys):
#                 return None
#         else:
#             return None
#     except (TypeError, ValueError):
#         return None

#     schematic = bool(plot.get("schematic"))
#     xlabel = (data.get("x_axis") or {}).get("label") or ""
#     ylabel = (data.get("y_axis") or {}).get("label") or ""
#     title = data.get("title") or ""

#     fig, ax = plt.subplots(figsize=(8, 6))
#     if kind == "bar":
#         hatches = ["//", "\\\\", "xx", "..", "++", "oo"]
#         bars = ax.bar(cats, vals, facecolor="white", edgecolor="black", linewidth=4,
#                       hatch=None)
#         for i, b in enumerate(bars):
#             b.set_hatch(hatches[i % len(hatches)])
#             ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
#                     f"{vals[i]:g}", ha="center", va="bottom",
#                     fontsize=22, fontweight="bold")
#         ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
#     else:
#         ax.plot(xs, ys, color="black", linewidth=7,
#                 marker=None if schematic else "s", markersize=16)
#         if schematic:
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.text(0.5, -0.18, "Schematic: shape only, no exact values",
#                     transform=ax.transAxes, ha="center", fontsize=16)

#     ax.set_xlabel(xlabel, fontsize=24, fontweight="bold")
#     ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
#     ax.set_title(title, fontsize=26, fontweight="bold", pad=20)
#     ax.tick_params(axis="both", labelsize=20, width=4, length=10)
#     for s in ("top", "right"):
#         ax.spines[s].set_visible(False)
#     for s in ("left", "bottom"):
#         ax.spines[s].set_linewidth(6)
#     fig.tight_layout()

#     out = []
#     for fmt in ("png", "svg", "pdf"):
#         buf = io.BytesIO()
#         fig.savefig(buf, format=fmt, dpi=150, facecolor="white")
#         out.append(buf.getvalue())
#     plt.close(fig)
#     return tuple(out)


# # ---------- Quiz ----------
# QUIZ_FOCUS = [
#     "reading exact values from the diagram",
#     "comparing two parts of the diagram (higher, lower, difference)",
#     "what the diagram means in a real-life situation",
#     "definitions and the underlying concept",
#     "simple calculations using the values in the diagram",
#     "common mistakes students make with this kind of diagram",
#     "what would change if a value changed",
# ]


# def make_quiz(data, lang, avoid=None):
#     focus = "; ".join(random.sample(QUIZ_FOCUS, 3))
#     avoid_txt = ""
#     if avoid:
#         avoid_txt = ("\nDo NOT repeat or rephrase any of these earlier questions:\n- " +
#                      "\n- ".join(avoid[-20:]))
#     prompt = f"""Create 5 NEW multiple-choice questions for a school student, in {lang}
# (Devanagari script if Hindi), based on this diagram data:
# {json.dumps(data)}
# Emphasise these angles this time: {focus}.{avoid_txt}
# Questions about the diagram must use only values present in the data, never invent numbers.
# Include at least 1 concept question. Each question has exactly 4 options.
# Put the correct answer at a random position (not always the same index).
# Return ONLY a JSON list. Each item: {{"question": str, "options": [4 strings],
# "answer_index": 0-3, "explanation": one short sentence}}.
# Plain text only, no markdown."""
#     result = parse_json(call_llm([prompt], as_json=True, temperature=1.0))
#     if isinstance(result, dict):
#         result = result.get("questions", [])
#     good = []
#     for q in result:
#         try:
#             if len(q["options"]) == 4 and 0 <= int(q["answer_index"]) <= 3:
#                 q["answer_index"] = int(q["answer_index"])
#                 good.append(q)
#         except (KeyError, TypeError, ValueError):
#             continue
#     return good


# st.set_page_config(page_title="DiagramVoice", page_icon="🔊", layout="wide")

# st.markdown("""
# <style>
# html, body, [class*="css"] { font-size: 18px; }
# #MainMenu, footer { visibility: hidden; }
# .block-container { padding-top: 1.5rem; max-width: 1250px; }
# .hero { background: linear-gradient(135deg, #1e3a8a, #2563eb); padding: 1.8rem 2rem;
#         border-radius: 20px; margin-bottom: 1.2rem; }
# .hero-title { color: #ffffff; font-size: 2.6rem; font-weight: 800; line-height: 1.1; }
# .hero-sub { color: #e0e7ff; font-size: 1.25rem; margin-top: .5rem; }
# .badges span { display: inline-block; background: rgba(255,255,255,.18); color: #fff;
#         padding: .3rem .8rem; border-radius: 999px; margin: .8rem .4rem 0 0; font-size: .95rem; }
# .step { background: #eef2ff; border-radius: 16px; padding: 1.1rem 1.2rem; color: #0f172a; }
# .step b { font-size: 1.15rem; }
# .stButton > button, .stDownloadButton > button {
#         font-size: 1.1rem; font-weight: 700; padding: .7rem 1.4rem; border-radius: 12px; }
# button:focus-visible, input:focus-visible, [role="radio"]:focus-visible {
#         outline: 4px solid #f59e0b !important; outline-offset: 2px; }
# </style>
# <div class="hero">
#   <div class="hero-title">🔊 DiagramVoice</div>
#   <div class="hero-sub">Hear any textbook graph or circuit. Ask your doubts by voice, in Hindi or English.</div>
#   <div class="badges"><span>Hindi + English</span><span>Voice in, voice out</span>
#   <span>Never invents numbers</span><span>Tactile-ready export</span><span>Free to use</span></div>
# </div>
# """, unsafe_allow_html=True)

# left, right = st.columns([1, 1.35], gap="large")

# # ======================= LEFT: upload and settings =======================
# with left:
#     st.subheader("1. Choose your diagram")
#     lang = st.radio("Language", ["English", "Hindi"], horizontal=True)
#     vc1, vc2 = st.columns(2)
#     vc1.selectbox("Voice accent (English only)", list(ACCENTS), key="accent")
#     vc2.radio("Speaking speed", ["Normal", "Slow"], key="speed", horizontal=True)
#     file = st.file_uploader("Upload a photo, image, or PDF of the diagram",
#                             type=["png", "jpg", "jpeg", "pdf"])

#     img = None
#     if file:
#         if file.name.lower().endswith(".pdf"):
#             import fitz  # PyMuPDF
#             pdf = fitz.open(stream=file.getvalue(), filetype="pdf")
#             page_no = 1
#             if len(pdf) > 1:
#                 page_no = st.number_input(
#                     f"This PDF has {len(pdf)} pages. Which page has the diagram?",
#                     min_value=1, max_value=len(pdf), value=1, step=1)
#             pix = pdf[int(page_no) - 1].get_pixmap(dpi=150)
#             img = Image.open(io.BytesIO(pix.tobytes("png")))
#         else:
#             img = Image.open(file)

#     if img is not None:
#         st.image(img, caption="Your diagram")
#         if st.button("🔊 Describe this diagram", type="primary", use_container_width=True):
#             try:
#                 with st.spinner("Reading the diagram..."):
#                     extra = f"""
# Also add a key "spoken_description": a spoken description in {lang} (Devanagari if Hindi),
# for a blind student, under 120 words, plain sentences, no markdown, in this order:
# kind of diagram, overall shape, key labelled parts, key values, what it means.
# Only mention uncertain items if the uncertain list is not empty; say "approximately" for estimated values."""
#                     data = parse_json(call_llm([shrink(img), EXTRACT_PROMPT + extra], as_json=True))
#                     text = data.get("spoken_description") or make_description(data, lang)
#                     tactile = make_tactile(data)
#                 st.session_state.update(data=data, text=text, lang=lang, tactile=tactile)
#                 st.session_state.pop("qa_cache", None)
#                 st.session_state.pop("quiz", None)
#                 st.session_state.pop("quiz_history", None)
#                 for k in [k for k in st.session_state if str(k).startswith("quiz_")
#                           and k != "quiz_history"]:
#                     st.session_state.pop(k, None)
#             except Exception as e:
#                 st.error(f"Something went wrong: {e}")

# # ======================= RIGHT: results =======================
# with right:
#     if "text" not in st.session_state:
#         st.subheader("How it works")
#         st.markdown("""
# <div class="step"><b>1. Upload</b><br>A photo, image, or PDF page of a graph, bar chart, or circuit.</div><br>
# <div class="step"><b>2. Listen</b><br>Get a clear spoken description in the same order every time.</div><br>
# <div class="step"><b>3. Ask and practise</b><br>Ask doubts by voice, download a tactile-ready version, and take a quiz.</div>
# """, unsafe_allow_html=True)
#     else:
#         st.subheader("2. Your results")
#         view = st.radio("Section",
#                         ["🔊 Listen", "✋ Tactile", "🎤 Ask", "📝 Quiz", "🔍 Under the hood"],
#                         horizontal=True, key="view", label_visibility="collapsed")
#         slang = st.session_state["lang"]

#         if view == "🔊 Listen":
#             st.write(st.session_state["text"])
#             st.audio(speak(st.session_state["text"], slang), format="audio/mp3")
#             if st.session_state["data"].get("uncertain"):
#                 st.warning("Not sure about: " +
#                            "; ".join(map(str, st.session_state["data"]["uncertain"])))
#             st.markdown("**Key facts**")
#             for f in st.session_state["data"].get("key_features", []):
#                 st.write("• " + str(f))

#         elif view == "✋ Tactile":
#             tac = st.session_state.get("tactile")
#             if tac:
#                 png, svg, pdf_bytes = tac
#                 st.image(png, caption="Simplified thick-line version for swell paper "
#                                       "or for a school to emboss")
#                 c1, c2 = st.columns(2)
#                 c1.download_button("⬇ Download SVG", svg, file_name="tactile_diagram.svg",
#                                    mime="image/svg+xml", use_container_width=True)
#                 c2.download_button("⬇ Download PDF", pdf_bytes, file_name="tactile_diagram.pdf",
#                                    mime="application/pdf", use_container_width=True)
#                 st.caption("A simplified redraw from the extracted data, ready for tactile "
#                            "printing. It is not a tactile device.")
#             else:
#                 st.info("A tactile version is available for graphs and bar charts. "
#                         "It is not available for this diagram type yet.")

#         elif view == "🎤 Ask":
#             voice = st.audio_input("🎤 Ask by voice")
#             typed = st.text_input("Or type your question")
#             question = typed
#             try:
#                 if voice is not None:
#                     audio_bytes = voice.getvalue()
#                     vid = hash(audio_bytes)
#                     if st.session_state.get("voice_id") != vid:
#                         with st.spinner("Listening..."):
#                             st.session_state["voice_q"] = transcribe(audio_bytes, slang)
#                         st.session_state["voice_id"] = vid
#                     if not typed:
#                         question = st.session_state["voice_q"]

#                 if question:
#                     cache = st.session_state.setdefault("qa_cache", {})
#                     key = (question, slang)
#                     if key not in cache:
#                         with st.spinner("Thinking..."):
#                             ans = call_llm([f"""You are a patient school tutor helping a blind student.
# Answer in {slang}. If the language is Hindi, write in Devanagari script.

# The student is looking at this diagram (extracted data):
# {json.dumps(st.session_state['data'])}

# RULES:
# 1. If the question is about THIS diagram (its values, labels, shape, which is highest, etc.),
#    answer ONLY from the diagram data. If the value is not in the data, say you are not sure.
#    Never invent numbers.
# 2. If the question is a general concept or doubt (for example "what is a bar chart",
#    "what is acceleration", "how do I find slope"), explain it from your own knowledge
#    in simple words, at school level, with one short everyday example.
#    Where useful, connect it to this diagram.
# 3. If the question is far outside school subjects, politely say you can only help with
#    study doubts related to this topic.
# 4. Speak in plain sentences only. No markdown, bullet points, or symbols.
#    Keep it under 100 words, because this will be read aloud.

# Question: {question}"""])
#                             cache[key] = (ans, None)
#                     ans, _ = cache[key]
#                     st.write(f"**Question:** {question}")
#                     st.write(f"**Answer:** {ans}")
#                     st.audio(speak(ans, slang), format="audio/mp3")
#             except Exception as e:
#                 st.error(f"Something went wrong: {e}")

#         elif view == "📝 Quiz":
#             if st.button("Create a quiz on this diagram", type="primary"):
#                 try:
#                     with st.spinner("Writing questions..."):
#                         history = st.session_state.get("quiz_history", [])
#                         quiz = make_quiz(st.session_state["data"], slang, avoid=history)
#                         st.session_state["quiz_history"] = history + [q["question"] for q in quiz]
#                     for k in [k for k in st.session_state if str(k).startswith("quiz_")
#                               and k not in ("quiz_history",)]:
#                         st.session_state.pop(k, None)
#                     st.session_state["quiz"] = quiz
#                     if not quiz:
#                         st.warning("Could not make a quiz this time. Please try again.")
#                 except Exception as e:
#                     st.error(f"Something went wrong: {e}")

#             quiz = st.session_state.get("quiz")
#             if quiz:
#                 if st.checkbox("🔊 Read the quiz aloud"):
#                     spoken = " ".join(
#                         f"Question {i + 1}. {q['question']}. Options: " +
#                         ". ".join(f"{'ABCD'[j]}, {o}" for j, o in enumerate(q["options"]))
#                         for i, q in enumerate(quiz))
#                     st.audio(speak(spoken, slang), format="audio/mp3")

#                 for i, q in enumerate(quiz):
#                     st.radio(f"Question {i + 1}. {q['question']}", q["options"],
#                              index=None, key=f"quiz_{i}")

#                 if st.button("Check my answers"):
#                     score = 0
#                     for i, q in enumerate(quiz):
#                         chosen = st.session_state.get(f"quiz_{i}")
#                         correct = q["options"][q["answer_index"]]
#                         if chosen == correct:
#                             score += 1
#                             st.success(f"Question {i + 1}: correct. {q.get('explanation', '')}")
#                         else:
#                             st.error(f"Question {i + 1}: the correct answer is {correct}. "
#                                      f"{q.get('explanation', '')}")
#                     st.subheader(f"Your score: {score} out of {len(quiz)}")

#         else:
#             st.caption("The AI first extracts structured data from the diagram, then every "
#                        "explanation is written from this data. This is why it does not invent numbers.")
#             st.json(st.session_state["data"])

# st.divider()
# st.caption("AI can make mistakes. Please check important answers with your teacher. "
#            "DiagramVoice · HackNova 2026 · Inclusive Technology")

