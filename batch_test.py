import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.5-flash-lite","gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-flash-latest"]

PROMPT = """You are describing a diagram for a blind student.
Return ONLY valid JSON with these keys:
- type: "graph", "circuit" or "other"
- title: string
- x_axis: {"label": string, "unit": string or null, "range": [min, max] or null}
- y_axis: same structure
- curves: list of {"shape": string, "slope": "positive"/"negative"/"zero"/"varying",
  "starts_at": string, "ends_at": string,
  "points": list of [x, y] ONLY if numbers are visible, else []}
- components: list of circuit parts and how they connect (for circuits, else [])
- key_features: list of short plain sentences explaining what the diagram means
- uncertain: list of things you could not read clearly, or numbers not shown.
RULES: Never invent numbers. If a value is not printed, put it in "uncertain"."""

os.makedirs("results", exist_ok=True)

for fname in sorted(os.listdir("test_images")):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img = Image.open(f"test_images/{fname}")
    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[img, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=30000),
                ),
            )
            with open(f"results/{fname}.json", "w", encoding="utf-8") as f:
                f.write(resp.text)
            print("OK  ", fname, model, flush=True)
            break
        except Exception as e:
            print("FAIL", fname, model, str(e)[:70], flush=True)