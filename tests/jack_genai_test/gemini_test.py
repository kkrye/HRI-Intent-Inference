from google import genai
from google.genai import types
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import time


drawing=False
ix,iy = -1,-1
def draw_circle(event, x, y, flags, param):
    global ix,iy,drawing
    pen_width = 10
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(scratch_pad,(x,y),pen_width,(0,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(scratch_pad,(x,y),pen_width,(0,0,0),-1)


if __name__ == "__main__":
    NUM_CYCLES = 3
    file_time = time.strftime("%m%d%y_%H%M%S", time.localtime())

    # Possible prompts
    full_prompt = """This is a partial drawing done by me. I need you to do three things
        1. Make a guess on what is trying to be drawn
        2. Propose a drawing subject that is diametrically opposite of your first answer
        3. Send back the drawing with only a few strokes add so that the drawing is more similar to your answer to question 2

        NOTE: make sure only minimal changes are made to the drawing also keep the strokes simple."""
    # Partial prompts
    predict_prompt = "This is a partial drawing done by me. Make a guess on what is trying to be drawn. NOTE: keep your response to just the answer and no additional words."
    opposite_prompt = "This is a partial drawing done by me. Propose a drawing subject that is diametrically opposite of the drawing. NOTE: keep your response to just the answer and no additional words."
    add_prompt = """This is a partial drawing done by me. Send back the drawing with only a few strokes add so that the drawing is more similar to RESPONSE.
                    NOTE: make sure only minimal changes are made to the drawing also keep the strokes simple."""

    # Initialize the Gemini client
    with open(".config_files/gemini-api.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
    client = genai.Client(api_key=GOOGLE_API_KEY)

    scratch_pad = np.full((1024, 1024, 3),255, dtype = np.uint8)

    window_name = "Draw Window (ENTER to accept, ESC to quit)"
    cv2.namedWindow(winname = window_name)
    cv2.setMouseCallback(window_name, draw_circle)
    end_flag = False
    for i in range(NUM_CYCLES):
        while True:
            cv2.imshow(window_name, scratch_pad)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or k == 13:
                end_flag = k == 27
                break
        cv2.imwrite(f"temp_files/{file_time}_pic_{i}a.png", scratch_pad)
        if end_flag:
            exit(0)

        # convert cv image into bytes
        success, encoded_image = cv2.imencode('.png', scratch_pad)
        if not success:
            raise IOError("Failed to encode NumPy array into PNG format.")

        # prompt the model
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[
                types.Part.from_bytes(
                    data=encoded_image.tobytes(),
                    mime_type="image/png"
                ),
                full_prompt
            ],
            config = types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                )
        )
        # Extract info from response
        text_output = ""
        for part in response.candidates[0].content.parts:
            if part.text:
                text_output+=f"Text Output: {part.text.strip()}"
            elif part.inline_data:
                text_output+=f"Image Output: Found image with mime type {part.inline_data.mime_type}"
                image_bytes = part.inline_data.data
                image = Image.open(BytesIO(image_bytes))
                image.save(f"temp_files/{file_time}_pic_{i}b.png")
        img_response = cv2.imread(f"temp_files/{file_time}_pic_{i}b.png")
        bitwise_and_img = cv2.bitwise_and(scratch_pad, img_response)
        ratio_img_kept = bitwise_and_img.mean() / scratch_pad.mean()
        text_output+=f"\nRatio of image kept the same: {ratio_img_kept:.2f}\n"
        # TODO: Add loop to reprompt if too little of the image was kept
        with open(f"temp_files/{file_time}_text_{i}.txt", "w") as f:
            f.write(text_output)
        print(text_output)

        scratch_pad = img_response.copy()