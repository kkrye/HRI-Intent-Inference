import torch
import torch.nn as nn


def create_gaussian_heatmap(coords, resolution=64, sigma=0.02):
    """
    Creates a differentiable heatmap from a sequence of normalized coordinates,
    treating the stroke as a series of connected line segments.
    Coords are expected to be in the normalized [-1, 1] space.

    Args:
        coords (torch.Tensor): Normalized coordinates [B, L, 2].
        resolution (int): HxW resolution of the output heatmap (e.g., 64).
        sigma (float): Standard deviation of the Gaussian kernel (controls smoothness/thickness).

    Returns:
        torch.Tensor: Heatmap tensor [B, 1, H, W].
    """
    device = coords.device
    B, L, _ = coords.shape

    # Check if there are enough points for segments (L >= 2)
    if L < 2:
        # If not enough points, return an empty heatmap (zero loss)
        return torch.zeros(B, 1, resolution, resolution, device=device)

    # 1. Create a coordinate grid for the output image (H x W)
    # Grid coordinates range from [-1, 1]
    xv, yv = torch.meshgrid([torch.linspace(-1, 1, resolution, device=device)] * 2, indexing='ij')
    grid = torch.stack([yv, xv], dim=-1) # [H, W, 2]

    # Grid expanded for broadcasting: [1, 1, H, W, 2]
    grid_expanded = grid.unsqueeze(0).unsqueeze(1)

    # 2. Define Line Segments (P1 and P2)
    P1 = coords[:, :-1].unsqueeze(2).unsqueeze(3) # Start points [B, L-1, 1, 1, 2]
    P2 = coords[:, 1:].unsqueeze(2).unsqueeze(3)  # End points [B, L-1, 1, 1, 2]

    # 3. Vector Calculation
    V = P2 - P1         # Segment vector (P1 -> P2) [B, L-1, 1, 1, 2]
    W = grid_expanded - P1 # Vector (P1 -> Grid Point) [B, L-1, H, W, 2]

    # V dot V (Squared length of the segment) [B, L-1, 1, 1]
    V_sq = torch.sum(V * V, dim=-1)

    # W dot V (Projection parameter numerator) [B, L-1, H, W]
    W_dot_V = torch.sum(W * V, dim=-1)

    # 4. Find the closest point on the segment
    # t is the scalar projection parameter for the infinite line
    # Clamp V_sq to prevent division by zero, but rely on geometric logic below
    t = W_dot_V / V_sq.clamp(min=1e-8)

    # Clamp t to [0, 1] to ensure the closest point is on the SEGMENT
    t_clamped = torch.clamp(t, 0.0, 1.0) # [B, L-1, H, W]

    # Closest Point (Pc) on the segment P1P2: Pc = P1 + t_clamped * V
    t_expanded = t_clamped.unsqueeze(-1) # [B, L-1, H, W, 1]
    Pc = P1 + t_expanded * V             # [B, L-1, H, W, 2]

    # 5. Calculate Squared Distance to the Closest Point
    # dist_sq = ||Grid Point - Closest Point||^2
    dist_sq = torch.sum((grid_expanded - Pc) ** 2, dim=-1) # [B, L-1, H, W]

    # 6. Calculate Gaussian weights
    # Gaussian formula: exp(-dist_sq / (2 * sigma^2))
    # Note: We sum over L-1 segments, not L points
    gaussian_maps = torch.exp(-dist_sq / (2 * sigma**2)) # [B, L-1, H, W]

    # 7. Aggregate all segments
    heatmap = torch.sum(gaussian_maps, dim=1) # [B, H, W]

    # 8. Normalize and Finalize
    # Clip values above 1.0
    heatmap = torch.clamp(heatmap, 0, 1)

    return heatmap.unsqueeze(1) # [B, 1, H, W]

class SequenceLossSmoothed(nn.Module):
    """
    Replaces point-wise coordinate loss with a smoothed image similarity loss (MSE
    on Gaussian heatmaps) while retaining the critical EOS classification loss.
    """
    def __init__(self, resolution=64, sigma=0.02):
        super(SequenceLossSmoothed, self).__init__()
        # Use MSE for image similarity (smoothed loss)
        self.img_criterion = nn.MSELoss()
        # Standard BCE for the classification task of terminating the stroke
        self.eos_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.resolution = resolution
        self.sigma = sigma

    def forward(self, pred_coords, pred_eos_logits, target_coords, target_eos):
        """
        Args:
            pred_coords: [B, L, 2] (Predicted normalized x, y)
            pred_eos_logits: [B, L] (Predicted EOS logit)
            target_coords: [B, L, 2] (Ground truth normalized x, y)
            target_eos: [B, L] (Ground truth EOS flag: 1.0 at EOS, 0.0 otherwise)
        """

        # 1. IMAGE/SHAPE LOSS (Smoothed Loss)
        # Use soft_mask_k in the heatmap generation
        pred_heatmap = create_gaussian_heatmap(pred_coords, self.resolution, self.sigma)
        target_heatmap = create_gaussian_heatmap(target_coords, self.resolution, self.sigma)

        # Calculate image similarity loss
        img_loss = self.img_criterion(pred_heatmap, target_heatmap)

        # 2. EOS Loss (Classification)
        eos_loss = self.eos_criterion(pred_eos_logits, target_eos)
        eos_loss = eos_loss.mean()

        # Returns the smoothed shape loss and the EOS loss
        return img_loss, eos_loss

class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()
        # Use L1 Loss (MAE) for coordinates for robustness against outliers
        self.coord_criterion = nn.L1Loss(reduction='none')
        self.eos_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_coords, pred_eos_logits, target_coords, target_eos):
        """
        Calculates loss for coordinates and EOS, using a mask to ignore padded steps.
        Args:
            pred_coords: [B, L, 2] (Predicted normalized x, y)
            pred_eos_logits: [B, L] (Predicted EOS logit)
            target_coords: [B, L, 2] (Ground truth normalized x, y)
            target_eos: [B, L] (Ground truth EOS flag: 1.0 at EOS, 0.0 otherwise)
        """

        # We only calculate coordinate loss up to the actual end of the stroke (or max length)
        # valid_mask is 1.0 for all steps *before* and *at* the EOS token (inclusive of EOS prediction)
        # Note: If target_eos has 1.0 at L, then the sequence length is L.

        # Mask calculation: Find the true length of each sequence
        # We use target_coords.sum(dim=2) to identify where padding starts (sum is 0).
        # We use 1.0 for valid points and 0.0 for padding.
        # Assuming padding in target_coords is 0, 0
        valid_length_mask = (target_coords.sum(dim=2) != -2).float()
        # This simple mask works if (0, 0) is not a common normalized coordinate.

        # 1. Coordinate Loss (MAE)
        coord_loss_unmasked = self.coord_criterion(pred_coords, target_coords).sum(dim=2) # [B, L]
        coord_loss = (coord_loss_unmasked * valid_length_mask).sum() / valid_length_mask.sum().clamp(min=1e-6)

        # 2. EOS Loss (BCE)
        # We train EOS loss over the entire sequence including padding,
        # but the coordinate loss only covers the actual stroke.
        eos_loss = self.eos_criterion(pred_eos_logits, target_eos).mean()

        return coord_loss, eos_loss

# --- TEACHER FORCING DECODER HELPER ---

def decode_lstm_sequence(model, h0, c0, context_features, text_context, target_coords):
    """
    Decodes the sequence using Teacher Forcing (passing ground truth coordinates
    as input to the next time step).

    Args:
        model: The StrokeGeneratorModel instance.
        h0, c0: Initial hidden states [1, B, H].
        context_features: [B, H*W, C] (Encoded image features).
        target_coords: [B, L, 2] (Target normalized coordinates used for input).

    Returns:
        pred_coords: [B, L, 2]
        pred_eos_logits: [B, L]
    """
    B, L, _ = target_coords.size()

    # Initialize output storage
    pred_coords = torch.zeros(B, L, 2).to(target_coords.device)
    pred_eos_logits = torch.zeros(B, L).to(target_coords.device)

    # LSTM states (h_t, c_t) are initially (h0[0], c0[0]) since h0/c0 are [1, B, H]
    hx, cx = h0.squeeze(0), c0.squeeze(0)

    # The first input (t=0) is the start token (0, 0)
    # We loop from t=0 to L-1 (L steps)
    input_coord = torch.zeros(B, 2).to(target_coords.device)

    for t in range(L):
        # 1. Attention (hx is the current hidden state [B, H])
        context_vector, _ = model.attention(context_features, hx) # [B, C]

        # 2. LSTM Input: Concatenate current coordinate and context vector
        lstm_input = torch.cat([input_coord, context_vector,text_context], dim=1).unsqueeze(1) # [B, 1, 2+C+H]

        # 3. LSTM Step: (h_t, c_t) = LSTM(input, (h_{t-1}, c_{t-1}))
        lstm_output, (hx_t, cx_t) = model.lstm(lstm_input, (hx.unsqueeze(0), cx.unsqueeze(0)))

        # Update states for next step
        hx, cx = hx_t.squeeze(0), cx_t.squeeze(0)

        # 4. Final Output: [B, 3] (x, y, EOS logit)
        output = model.output_layer(lstm_output.squeeze(1))

        # Store predictions
        next_coord_raw = output[:, :2] # Raw (x, y) prediction
        pred_coords[:, t, :] = torch.tanh(next_coord_raw) # Bounded (x, y)
        # pred_coords[:, t, :] = output[:, :2]
        pred_eos_logits[:, t] = output[:, 2]

        # Teacher Forcing: The next input coordinate is the ground truth for this step
        if t < L - 1:
            input_coord = target_coords[:, t, :] # Use the ground truth coordinate

    return pred_coords, pred_eos_logits

class TVLoss(nn.Module):
    """
    Calculates the Total Variation Loss to encourage spatial smoothness.
    Penalizes the difference between adjacent pixels.
    """
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        # x is the model output (e.g., [B, C, H, W] in the [0, 1] range)

        # 1. Horizontal variation (difference between pixel (i, j) and (i, j+1))
        # We slice the tensor to align pixels for subtraction.
        # This calculates the differences for all but the last column.
        horiz_diff = x[:, :, :, :-1] - x[:, :, :, 1:]

        # 2. Vertical variation (difference between pixel (i, j) and (i+1, j))
        # This calculates the differences for all but the last row.
        vert_diff = x[:, :, :-1, :] - x[:, :, 1:, :]

        # Calculate the L2-norm squared for horizontal and vertical differences
        # We use squaring (L2) instead of absolute value (L1) as it is differentiable
        # and often results in better optimization.
        tv_loss = torch.sum(horiz_diff.pow(2)) + torch.sum(vert_diff.pow(2))

        # Scale by the weight
        return self.weight * tv_loss

class DiceLoss(nn.Module):
    """
    Computes 1 - Dice Score, which is (1 - 2*Intersection / (A + B)).
    Assumes input 'pred' is the probability output (after Sigmoid).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten tensors to operate on pixels, not batches/channels
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate Intersection (True Positives)
        intersection = (prediction * target).sum()

        # Calculate Union (Sum of all predicted and target pixels)
        dice_sum = prediction.sum() + target.sum()

        # Dice Score: 2 * Intersection / Union
        dice_score = (2. * intersection + self.smooth) / (dice_sum + self.smooth)

        # Dice Loss: 1 - Dice Score
        return 1. - dice_score
