import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from transformers import BertTokenizer, BertModel

def save_model_weights_and_args(model,path,logs=None):
    # HACK
    arg_dict = {k: v for k, v in model.__dict__.items() if k[0] != '_' and k!='tokenizer'}
    torch.save({'model_kargs': arg_dict,'model_state_dict': model.state_dict(),'logs':logs}, path)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        # Only add dropout if the rate is non-zero
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            # Note: Dropout2d is the correct choice for Conv layers

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

# --- 2. The Conditional U-Net Model ---
class ArtistModel(nn.Module):
    def __init__(self, img_channels=1, base_c=64,depth = 4, dropout_rate=.2, bert_model_name = 'bert-base-uncased',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.depth=depth
        self.device = device
        # Define channel sizes
        self.c_levels = [base_c*(2**i) for i in range(depth)]

        # Create Encoder
        self.encoders = nn.ModuleList([ConvBlock(img_channels if d==0 else self.c_levels[d-1], self.c_levels[d],dropout_rate=dropout_rate) for d in range(depth)])
        self.poll = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. BOTTLENECK (Feature Fusion Point)
        # The input to the bottleneck comes from the image (self.c_levels[-1]) plus the expanded language embedding (embed_dim).
        self.bottleneck = ConvBlock(self.c_levels[-1] + self.embed_dim, self.c_levels[-1],dropout_rate=dropout_rate*2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in range(depth-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(self.c_levels[d], self.c_levels[d-1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(self.c_levels[d-1]+self.c_levels[d], self.c_levels[d-1]))  # times 2 for skip connection

        self.upconvs.append(nn.ConvTranspose2d(self.c_levels[0], self.c_levels[0], kernel_size=2, stride=2))
        self.decoders.append(ConvBlock(self.c_levels[0]*2, self.c_levels[0]))

        self.out_conv = nn.Conv2d(self.c_levels[0] , img_channels, kernel_size=1)

        self.to(self.device)

    @staticmethod
    def to_bw_img(img:torch.Tensor):
        return (img > 0.5).float() * 255

    def to_float_img(self,img:torch.Tensor):
        return img.unsqueeze(1).float().to(self.device) / 255.0

    def forward(self, x, text_input):        # x: (B, 1, 256, 256), lang_embedding: (B, E_DIM)
        c_inner = []
        for enc in self.encoders:
            x = enc(x)
            c_inner.append(x)
            x = self.poll(x)
        encoded_img = x

        tokenized_input = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Ensure BERT tensors are on the same device as the image
        tokenized_input = {k: v.to(x.device) for k, v in tokenized_input.items()}

        # Get BERT output (no gradient tracking for initial BERT output)
        bert_output = self.bert_model(**tokenized_input)

        # Extract the contextual embedding for the whole sentence (the [CLS] token)
        # embedding_vector shape: (B, EMBEDDING_DIM=768)
        embedding_vector = bert_output.last_hidden_state[:, 0, :]

        # a. Prepare the Language Embedding for Concatenation
        # Reshape language vector to match feature map dimensions
        B, C, H_f, W_f = encoded_img.shape
        # Expand: (B, E_DIM) -> (B, E_DIM, 1, 1) -> (B, E_DIM, H_f, W_f)
        lang_feature_map = embedding_vector[:, :, None, None].expand(B, self.embed_dim, H_f, W_f)
        # b. Concatenate Image Features and Language Features
        fused_features = torch.cat([encoded_img, lang_feature_map], dim=1) # (B, 512 + E_DIM, 16, 16)
        # c. Bottleneck Convolution
        b = self.bottleneck(fused_features) # (B, 512, 16, 16)

        # --- 3. DECODER ---
        for i in range(self.depth):
            u = self.upconvs[i](b)
            skip_idx = self.depth - 1 - i
            c = c_inner[skip_idx]
            # CROP/PAD SAFETY CHECK (Still important)
            if u.shape[-2:] != c.shape[-2:]:
                diffY = c.size()[2] - u.size()[2]
                diffX = c.size()[3] - u.size()[3]
                c = c[:, :, diffY//2:c.size()[2] - (diffY - diffY//2), diffX//2:c.size()[3] - (diffX - diffX//2)]

            u = torch.cat([u, c], dim=1)
            b = self.decoders[i](u)

        # --- 4. OUTPUT ---
        output_image = torch.sigmoid(self.out_conv(b))

        return output_image

# --- Attention Mechanism for LSTM Context ---
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        # W_h: Projects encoder features
        self.W_h = nn.Linear(encoder_dim, decoder_dim, bias=False)
        # W_s: Projects decoder hidden state
        self.W_s = nn.Linear(decoder_dim, decoder_dim, bias=False)
        # V: Projects combined energy to a single score
        self.V = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_features, decoder_hidden):
        # encoder_features: [B, H*W, C]
        # decoder_hidden: [B, D]

        # 1. Expand decoder state for broadcast addition: [B, 1, D]
        hidden_expanded = decoder_hidden.unsqueeze(1)

        # 2. Calculate Energy: tanh(W_h*Context + W_s*Hidden) -> [B, H*W, D]
        # We need to reshape encoder_features for the linear layer
        energy = torch.tanh(self.W_h(encoder_features) + self.W_s(hidden_expanded))

        # 3. Calculate Attention Scores: V * energy -> [B, H*W]
        scores = self.V(energy).squeeze(2)

        # 4. Normalize Scores (Softmax over spatial locations)
        alphas = F.softmax(scores, dim=1) # [B, H*W]

        # 5. Compute Context Vector: Weighted sum of context features -> [B, C]
        context_vector = (alphas.unsqueeze(2) * encoder_features).sum(dim=1)

        return context_vector, alphas

class StrokeModel(nn.Module):
    def __init__(self, max_stroke_len, base_c=64, depth=3, hidden_size=512,lang_hidden_size=128, dropout_p=0.2, img_length=256,
                 bert_model_name = 'bert-base-uncased',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        img_channels = 1
        self.img_len = img_length
        self.hidden_size = hidden_size
        self.lang_hidden_size = lang_hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.bert_projection = nn.Linear(self.embed_dim, self.lang_hidden_size)

        # --- U-NET ENCODER (Image Feature Extraction) ---
        self.depth = depth
        self.c_levels = [base_c * (2**d) for d in range(depth)]

        # Encoders are standard ConvBlocks with downsampling
        self.encoders = nn.ModuleList([
            # Assuming you reuse the ConvBlock from the previous U-Net structure
            ConvBlock(img_channels if d == 0 else self.c_levels[d - 1], self.c_levels[d], dropout_rate=dropout_p)
            for d in range(depth)
        ])

        # Downsampling layers
        self.downs = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(depth)
        ])

        # Bottleneck (Feature Fusion with Text/BERT, assuming BERT features are [768, 1, 1])
        # We need to broadcast and concatenate the BERT features to the image features
        self.bottleneck_c = self.c_levels[-1]
        self.bottleneck_conv = ConvBlock(self.bottleneck_c + self.embed_dim, self.bottleneck_c, dropout_rate=dropout_p + 0.1)

        # --- HEADS (Split into 2 tasks) ---

        # 1. PARAMETER HEAD (MLP for Transformation Parameters)
        # Output: [mu_x, mu_y, S, cos(theta), sin(theta)] = 5 parameters
        # Note: Using sin/cos is better for regression than a raw angle theta
        # Input features: Flattened bottleneck features (bottleneck_c * 4*4 if last HxW=4)
        final_h_w = int(self.img_len // (2 ** self.depth))
        total_init_size = self.bottleneck_c * final_h_w * final_h_w + self.embed_dim
        self.param_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_init_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 5) # 5 outputs: mu_x, mu_y, S, cos(theta), sin(theta)
        )

        # 2. SEQUENCE DECODER HEAD (LSTM for Normalized Coordinates)
        self.lstm_hidden_size = hidden_size

        # Initial hidden state (h0, c0) from bottleneck features
        # Bottleneck features are projected down to the size of the LSTM hidden state
        self.h_init = nn.Linear(total_init_size, hidden_size)
        self.c_init = nn.Linear(total_init_size, hidden_size)

        # Attention Mechanism (Context from bottleneck, Hidden from LSTM)
        self.attention = Attention(encoder_dim=self.bottleneck_c, decoder_dim=hidden_size)

        # LSTM Decoder (Input: previous coordinate (2) + context vector (hidden_size))
        self.lstm = nn.LSTM(input_size=2 + self.bottleneck_c + self.lang_hidden_size,
                            hidden_size=hidden_size,
                            batch_first=True)

        # Output layer: 2 coordinates (x, y) + 1 EOS logit
        self.output_layer = nn.Linear(hidden_size, 3)

        # Store max length for inference
        self.max_stroke_len = max_stroke_len
        self.device = device
        self.to(device)

    def forward(self, img_x, text_input):
        # 1. ENCODER PASS
        skips = []
        x = img_x
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.downs[i](x)

        tokenized_input = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Ensure BERT tensors are on the same device as the image
        tokenized_input = {k: v.to(x.device) for k, v in tokenized_input.items()}

        # Get BERT output (no gradient tracking for initial BERT output)
        bert_output = self.bert_model(**tokenized_input)

        # Extract the contextual embedding for the whole sentence (the [CLS] token)
        # embedding_vector shape: (B, EMBEDDING_DIM=768)
        embedding_vector = bert_output.last_hidden_state[:, 0, :]
        projected_bert = self.bert_projection(embedding_vector)

        # 2. BOTTLENECK (Feature Fusion)
        # Bert embeds are typically [B, 768]. Reshape to [B, 768, 1, 1] and tile.
        bert_broadcast = embedding_vector.view(embedding_vector.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, bert_broadcast], dim=1)
        context : torch.Tensor = self.bottleneck_conv(x)

        # 3. SPLIT HEADS

        # Parameter Head
        flat_features = context.view(context.size(0), -1)
        flat_features_with_embedd = torch.cat([flat_features, embedding_vector], dim=1)
        raw_params = self.param_head(flat_features_with_embedd)

        # 4. APPLY BOUNDING ACTIVATIONS (CRITICAL STEP)
        pred_params = torch.empty_like(raw_params)

        # Indices 0, 1, 2 are Position (mu_x, mu_y) and Scale (S) -> Targets [0, 1] -> Sigmoid
        pred_params[:, 0:3] = torch.sigmoid(raw_params[:, 0:3])

        # Indices 3, 4 are Rotation (cos, sin) -> Targets [-1, 1] -> Tanh
        pred_params[:, 3:5] = torch.tanh(raw_params[:, 3:5])

        # LSTM Initial States
        h0 = torch.tanh(self.h_init(flat_features_with_embedd)).unsqueeze(0) # [1, B, H]
        c0 = torch.tanh(self.c_init(flat_features_with_embedd)).unsqueeze(0) # [1, B, H]

        # Context Features for Attention (Reshaped: [B, H*W, C])
        attn_context_features = context.view(context.size(0), self.bottleneck_c, -1).transpose(1, 2)

        # Returns the predicted parameters and the context needed for the Sequence Decoding loop
        return pred_params, h0, c0, attn_context_features, projected_bert