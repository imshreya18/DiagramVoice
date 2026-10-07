import os, json, io
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from gtts import gTTS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
          "gemini-3.5-flash", "gemini-flash-latest"]

EXTRACT_PROMPT = """You are describing a diagram for a blind student.
Return ONLY valid JSON with keys: type ("graph"/"circuit"/"other"), title,
x_axis {label, unit, range}, y_axis {label, unit, range},
curves [{shape, slope, starts_at, ends_at, points}],
components [circuit parts and connections, else []],
key_features [short plain sentences], uncertain [things not clearly readable].
RULES: Never invent numbers. If a value is not printed, put it in "uncertain"."""

def call_llm(contents, as_json=False):
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if as_json else None,
        http_options=types.HttpOptions(timeout=30000),
    )
    for model in MODELS:
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=cfg).text
        except Exception:
            continue
    raise RuntimeError("The AI service is busy. Please try again in a minute.")

def make_description(data, lang):
    prompt = f"""Write a spoken description of this diagram for a blind student,
in {lang}. Use this fixed order: what kind of diagram it is, its overall shape,
key labelled parts, key values, what it means. Mention anything listed as
uncertain honestly. Plain sentences only, no markdown, under 120 words.
Diagram data: {json.dumps(data)}"""
    return call_llm([prompt])

def make_audio(text, lang):
    buf = io.BytesIO()
    gTTS(text=text, lang="hi" if lang == "Hindi" else "en").write_to_fp(buf)
    return buf.getvalue()

st.set_page_config(page_title="DiagramVoice", page_icon="🔊")
st.title("🔊 DiagramVoice")
st.write("Upload a textbook graph or circuit and hear it explained.")

lang = st.radio("Language", ["English", "Hindi"], horizontal=True)
file = st.file_uploader("Upload a diagram", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Your diagram")

    if st.button("Describe this diagram"):
        try:
            with st.spinner("Reading the diagram..."):
                data = json.loads(call_llm([img, EXTRACT_PROMPT], as_json=True))
                text = make_description(data, lang)
                audio = make_audio(text, lang)
            st.session_state.update(data=data, text=text, audio=audio)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if "text" in st.session_state:
    st.subheader("Description")
    st.write(st.session_state["text"])
    st.audio(st.session_state["audio"], format="audio/mp3")

    if st.session_state["data"].get("uncertain"):
        st.warning("Not sure about: " +
                   "; ".join(map(str, st.session_state["data"]["uncertain"])))

    with st.expander("Structured data (JSON)"):
        st.json(st.session_state["data"])

    q = st.text_input("Ask a question about this diagram")
    if q:
        ans = call_llm([f"""Answer briefly in {lang} for a blind student.
Use ONLY this diagram data. If it isn't in the data, say you are not sure.
Data: {json.dumps(st.session_state['data'])}
Question: {q}"""])
        st.write(ans)