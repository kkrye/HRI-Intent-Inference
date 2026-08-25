from mistralai import Mistral
import base64
import cv2
import PIL
import os
import numpy as np



drawing=False
ix,iy = -1,-1
def draw_circle(event, x, y, flags, param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(scratch_pad,(x,y),5,(0,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(scratch_pad,(x,y),5,(0,0,0),-1)


if __name__ == "__main__":
    full_prompt = """This is a partial drawing done by me. I need you to do three things
        1. Make a guess on what is trying to be drawn
        2. Propose a drawing subject that is diametrically opposite of your first answer
        3. Send back the drawing with only a few strokes add so that the drawing is more similar to your answer to question 2

        NOTE: make sure only minimal changes are made to the drawing also keep the strokes simple."""

    predict_prompt = "This is a partial drawing done by me. Make a guess on what is trying to be drawn. NOTE: keep your response to just the answer and no additional words."
    opposite_prompt = "This is a partial drawing done by me. Propose a drawing subject that is diametrically opposite of the drawing. NOTE: keep your response to just the answer and no additional words."
    add_prompt = """This is a partial drawing done by me. Send back the drawing with only a few strokes add so that the drawing is more similar to RESPONSE.
                    NOTE: make sure only minimal changes are made to the drawing also keep the strokes simple."""

    with open(".config_files/mistral-api.txt") as f:
        API_KEY = f.read().strip()
    client = Mistral(api_key=API_KEY)
    image_agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="Image Generation Agent",
        description="Agent used to generate images.",
        instructions="Use the image generation tool when you have to create images.",
        tools=[{"type": "image_generation"}],
        completion_args={
            "temperature": 0.3,
            "top_p": 0.95,
        }
    )
    import mistralai.models as models
    models.ConversationInputs
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,.temp_files/pic_0a.png"
                }
            ]
        }
    ]
    response = client.beta.conversations.start(
        agent_id=image_agent.id,
        inputs=messages
    )
    print(response)
    # model = "pixtral-12b-2409"

    # scratch_pad = np.full((500, 500, 3),255, dtype = np.uint8)

    # window_name = "Draw Window"
    # cv2.namedWindow(winname = window_name)
    # cv2.setMouseCallback(window_name, draw_circle)
    # end_flag = False
    # for i in range(2):
    #     while True:
    #         cv2.imshow(window_name, scratch_pad)
    #         k = cv2.waitKey(10) & 0xFF
    #         if k == 27 or k == 13:
    #             end_flag = k == 27
    #             break
    #     cv2.destroyAllWindows()
    #     cv2.imwrite(f"temp_files/pic_{i}a.png", scratch_pad)
    #     if end_flag:
    #         exit(0)

    #     response = client.models.generate_content(
    #         model="gemini-2.5-flash-image-preview",
    #         contents=[
    #             types.Part.from_bytes(
    #                 data=scratch_pad.tobytes(),
    #                 mime_type="image/png"
    #             ),
    #             full_prompt
    #         ],
    #     )
    #     b64_out = response.candidates[0].content.parts[0].data
    #     print(response.text.strip())
    #     cv2.imwrite(f"temp_files/pic_{i}b.png", base64.b64decode(b64_out))
    #     scratch_pad = cv2.imdecode(np.frombuffer(base64.b64decode(b64_out), np.uint8), cv2.IMREAD_COLOR)

    # # Read local image
    # with open("input_photo.jpg", "rb") as f:
    #     img_bytes = f.read()

    # # Using SDK helper to attach bytes as a part

    # # The returned image will typically be in response.candidates[].content.parts with base64-encoded data
    # # Decode and save (pseudo-access shown; check SDK response structure)
    # b64_out = response.candidates[0].content.parts[0].data  # example path
    # with open("edited.jpg","wb") as out:
    #     out.write(base64.b64decode(b64_out))
