import os, json, io, re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from gtts import gTTS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
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


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def make_description(data, lang):
    prompt = f"""Write a spoken description of this diagram for a blind student,
in {lang}. Use this fixed order: what kind of diagram it is, its overall shape,
key labelled parts, key values, what it means. Only if the uncertain list is not empty, 
mention those items honestly. If it is empty, do not say anything about uncertainty. Plain sentences only, no markdown, under 120 words.
If a value is listed as uncertain or estimated, say "approximately" when you speak it.
Diagram data: {json.dumps(data)}"""
    return call_llm([prompt])


def make_audio(text, lang):
    buf = io.BytesIO()
    gTTS(text=text, lang="hi" if lang == "Hindi" else "en").write_to_fp(buf)
    return buf.getvalue()

def transcribe(audio_bytes, lang):
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    return call_llm([part, f"Transcribe this spoken question exactly. The language is probably {lang}. Return only the text."]).strip()

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
                data = parse_json(call_llm([img, EXTRACT_PROMPT], as_json=True))
                text = make_description(data, lang)
                audio = make_audio(text, lang)
            st.session_state.update(data=data, text=text, audio=audio, lang=lang)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if "text" in st.session_state:
    st.subheader("Description")
    st.write(st.session_state["text"])
    st.audio(st.session_state["audio"], format="audio/mp3")

    if st.session_state["data"].get("uncertain"):
        st.warning("Not sure about: " +
                   "; ".join(map(str, st.session_state["data"]["uncertain"])))

    st.subheader("Ask a question")
    voice = st.audio_input("🎤 Ask by voice")
    typed = st.text_input("Or type your question")

    question = typed
    try:
        if voice is not None:
            audio_bytes = voice.getvalue()
            vid = hash(audio_bytes)
            if st.session_state.get("voice_id") != vid:
                with st.spinner("Listening..."):
                    st.session_state["voice_q"] = transcribe(audio_bytes, st.session_state["lang"])
                st.session_state["voice_id"] = vid
            if not typed:
                question = st.session_state["voice_q"]

        if question:
            cache = st.session_state.setdefault("qa_cache", {})
            key = (question, st.session_state["lang"])
            if key not in cache:
                with st.spinner("Thinking..."):
                                        ans = call_llm([f"""You are a patient school tutor helping a blind student.
Answer in {st.session_state['lang']}. If the language is Hindi, write in Devanagari script.

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
                cache[key] = (ans, make_audio(ans, st.session_state["lang"]))
            ans, ans_audio = cache[key]
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {ans}")
            st.audio(ans_audio, format="audio/mp3")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

    with st.expander("Structured data (JSON)"):
        st.json(st.session_state["data"])