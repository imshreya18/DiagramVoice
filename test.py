import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
img = Image.open("test_images/barchart.png")

prompt = """You are describing a diagram for a blind student.
Return ONLY valid JSON, no markdown, with these keys:
- type: "graph", "circuit" or "other"
- title: string
- x_axis: {"label": string, "unit": string or null, "range": [min, max] or null}
- y_axis: same structure
- curves: list of {"shape": string, "slope": "positive"/"negative"/"zero"/"varying",
  "starts_at": string, "ends_at": string,
  "points": list of [x, y] ONLY if numbers are visible, else []}
- key_features: list of short plain sentences explaining what the graph means
- uncertain: list of things you could not read clearly, or numbers that are not shown.
RULES: Never invent numbers. If a value is not printed on the image,
put it in "uncertain" instead of guessing."""

# resp = client.models.generate_content(
#     model="gemini-3.8-flash",
#     contents=[img, prompt],
# )
# print(resp.text)

import time
from google.genai import types

models_to_try = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]

for name in models_to_try:
    for attempt in range(2):
        try:
            print(f"Trying {name} (attempt {attempt + 1})...", flush=True)
            resp = client.models.generate_content(
                model=name,
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=30000)
                ),
            )
            print("SUCCESS with", name)
            print(resp.text)
            raise SystemExit
        except SystemExit:
            raise
        except Exception as e:
            print("Failed:", str(e)[:150], flush=True)
            time.sleep(2)