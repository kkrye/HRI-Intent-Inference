import torch
import numpy as np

from .models import StrokeModel
from .data_proccessing import transform_stroke


@torch.no_grad()
def inference_decode_stroke(model:StrokeModel, h0, c0, context_features,text_context, eos_threshold=0.5):
    """
    Generates a normalized stroke sequence using the LSTM, feeding its own
    predicted coordinate back as input (autoregressive decoding).
    """
    device = h0.device
    batch_size = h0.size(1)

    # Initialize output storage
    predicted_coords = []

    # LSTM states (h_t, c_t)
    hx, cx = h0.squeeze(0), c0.squeeze(0) # [B, H]

    # Initial input token is (0, 0)
    input_coord = torch.zeros(batch_size, 2).to(device) # [B, 2]

    # EOS status tracker
    is_eos = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for t in range(model.max_stroke_len):
        # 1. Attention: Get context vector based on current hidden state
        context_vector, _ = model.attention(context_features, hx) # [B, C]

        # 2. LSTM Input: Concatenate last predicted coordinate and context vector
        lstm_input = torch.cat([input_coord, context_vector,text_context], dim=1).unsqueeze(1) # [B, 1, 2+C+H]

        # 3. LSTM Step
        lstm_output, (hx_t, cx_t) = model.lstm(lstm_input, (hx.unsqueeze(0), cx.unsqueeze(0)))

        # Update states for next step
        hx, cx = hx_t.squeeze(0), cx_t.squeeze(0)

        # 4. Final Output: [B, 3] (x, y, EOS logit)
        output = model.output_layer(lstm_output.squeeze(1))

        next_coord_raw = output[:, :2] # Raw (x, y) prediction
        pred_xy = torch.tanh(next_coord_raw) # Bounded (x, y)
        # pred_xy = output[:, :2]
        pred_eos_logit = output[:, 2]

        # 5. Check EOS condition
        pred_eos_prob = torch.sigmoid(pred_eos_logit)
        newly_finished = (pred_eos_prob > eos_threshold) & (~is_eos)

        # Update EOS status: Once finished, keep it finished
        is_eos = is_eos | newly_finished

        # Apply a mask to prevent new coordinates from being added once EOS is reached
        if is_eos.all():
            break

        # Store predicted normalized coordinate for all steps (including steps after EOS)
        predicted_coords.append(pred_xy)

        # 6. Autoregressive step: The current prediction is the input for the next step
        # If a stroke is finished, the input for the next step should be fixed (e.g., 0)
        # to avoid generating junk, but we continue looping to fill max_len
        # or if other strokes in the batch are still generating.
        input_coord = pred_xy

    # Stack results: [B, N_steps, 2]
    if not predicted_coords:
        return torch.empty(batch_size, 0, 2).to(device)

    return torch.stack(predicted_coords, dim=1) # [B, L_actual, 2]

def run_inference_walkthrough(model :StrokeModel, img_x, labels):
    # Set model to evaluation mode
    model.eval()

    # --- STEP 1: Run Encoder and Parameter Head ---
    print("Step 1/4: Running Encoder and Parameter Head...")
    # This gets the transformation parameters and the initial LSTM state
    pred_params, h0, c0, context_features, text_context = model(img_x, labels)

    # --- STEP 2: Sequence Decoding ---
    print("Step 2/4: Running Autoregressive Sequence Decoder...")

    # The output is [B, L_actual, 2], containing the predicted normalized (x, y) coordinates
    # We set a max length and an EOS probability threshold
    pred_normed_coords_tensor = inference_decode_stroke(
        model,
        h0,
        c0,
        context_features,
        text_context,
        eos_threshold=0.7 # Often a high threshold is needed for clean breaks
    )

    print(f"   -> Predicted normalized sequence length: {pred_normed_coords_tensor.size(1)} points.")

    # Convert to numpy for utility function and un-normalization
    pred_params_np = pred_params.squeeze(0).cpu().detach().numpy()
    pred_normed_coords_np = pred_normed_coords_tensor.squeeze(0).cpu().numpy()


    # --- STEP 3: Preparing Data for Reconstruction (Numpy/Utility) ---
    print("Step 3/4: Preparing predicted data for reconstruction...")

    # A. Recreate the Padded Stroke for the Utility Function
    # The reconstruction function needs a full 'padded_stroke' array and an 'eos_arr'
    N_pred = pred_normed_coords_np.shape[0]

    # Pad to max_stroke_len with -1 (as in raw_stroke_to_normed)
    pad_len = model.max_stroke_len - N_pred
    padded_stroke_pred = np.pad(pred_normed_coords_np, ((0, pad_len), (0, 0)), constant_values=-1)

    # B. Create the EOS array (set EOS flag at the predicted length N_pred)
    eos_arr_pred = np.zeros(model.max_stroke_len, dtype=np.float32)
    if N_pred > 0:
        # Set EOS flag at the last predicted point index
        eos_arr_pred[N_pred - 1] = 1.0

    # --- STEP 4: Reconstruction (Using the utility function from the Canvas) ---
    print("Step 4/4: Reconstructing stroke geometry...")

    # Call the utility function with the predicted parameters and normalized coordinates
    reconstructed_stroke = transform_stroke(
        pred_params_np,
        padded_stroke_pred,
        eos_arr_pred
    )

    print(f"   -> Final reconstructed stroke shape (N x 2): {reconstructed_stroke.shape}")
    print("Inference complete. The reconstructed stroke is ready for drawing.")
    return reconstructed_stroke, pred_params_np,padded_stroke_pred,eos_arr_pred

