## Current State

- Messing around with VLMs
    - Only Gemini works and only the model `"gemini-2.0-flash-preview-image-generation"`
    - Seems to kinda keep the drawing especially after more than one iteration
        - There is a check to see how much of the given image is keep but currently it is only recorded
        - **Next Steps**: Add a loop to reprompt the model if the image is too far off

### Running

`python jack_test/gemini_test.py`

