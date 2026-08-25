from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client() # define your API key locally with the environment variable GEMINI_API_KEY.

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Create a picture of a simple sketch of a dragon in ten strokes or less", # TODO: how to make it actually output an image??
    # config=types.GenerateContentConfig(
    #     thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking - takes longer if thinking but may be necessary for more detailed responses?
    # ),
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image.png") # TODO: save in SVG format