import torch
import torch.nn as nn
import time
import os

from data_proccessing import build_doodle_dataset, to_device
from models import StrokeModel, save_model_weights_and_args, ArtistModel
from losses import SequenceLossSmoothed, DiceLoss, TVLoss, decode_lstm_sequence

def build_train_artist(train_dataloader,test_dataloader):
    # Build ArtistModel
    model_depth = 3
    learning_rate = 1e-3
    white_ratio = .95
    pos_weight_value = (1-white_ratio) / white_ratio
    num_epochs = 3
    lambda_bin = 0.01
    lambda_dice = 2  # Weight for Dice Loss (Good starting point: 0.05 to 0.2)
    lambda_tv = 1   # Weight for TV Loss (Start small: 0.005 to 0.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = ArtistModel(depth=model_depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_bse = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))
    criterion_dice = DiceLoss()
    criterion_tv = TVLoss(weight=1.0)
    #  Train ArtistModel
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (partial_imgs, labels, next_strokes) in enumerate(train_dataloader):
            partial_imgs = model.to_float_img(partial_imgs)
            next_strokes = model.to_float_img(next_strokes)
            optimizer.zero_grad()
            outputs = model(partial_imgs, labels)
            loss_bce  = criterion_bse(outputs, next_strokes)
            loss_dice = criterion_dice(outputs, next_strokes)
            loss_tv = criterion_tv(outputs)
            loss_binary = 4 * torch.mean(outputs * (1 - outputs))
            total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for partial_imgs, labels, next_strokes in test_dataloader:
                partial_imgs = model.to_float_img(partial_imgs)
                next_strokes = model.to_float_img(next_strokes)
                outputs = model(partial_imgs, labels)
                loss_bce  = criterion_bse(outputs, next_strokes)
                loss_dice = criterion_dice(outputs, next_strokes)
                loss_tv = criterion_tv(outputs)
                loss_binary = 4 * torch.mean(outputs * (1 - outputs))
                total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
                val_loss += total_loss.item()
        val_loss /= len(test_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

def build_stroke_model(max_len,**kwargs):
    model = StrokeModel(max_stroke_len=max_len,**kwargs)
    print(f"Model initialized on device: {model.device}")
    return model

def train_stroke_model(model:StrokeModel,train_dataloader,test_dataloader,seq_loss_res=64, seq_loss_sig=0.025,lambda_seq_coord = 10.0,lambda_seq_eos = 0.25,lambda_params = .01 ,num_epochs:int=4,learning_rate=1e-4,save_model=True,report_rate=None):
    if save_model:
        print(f"Will save at {os.path.dirname(os.path.realpath(__file__))}/saved_models")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- LOSS FUNCTIONS ---
    # criterion_sequence = SequenceLoss()
    criterion_sequence = SequenceLossSmoothed(seq_loss_res, seq_loss_sig)
    criterion_params = nn.MSELoss()
    if report_rate is None:
        report_rate = int(len(train_dataloader)//10)
    logging_str=""
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0; running_p = 0.0; running_c = 0.0; running_e = 0.0

        # --- SIMULATED LOOP START (Replace with actual dataloader iteration) ---
        for i, batch in enumerate(train_dataloader):
            # 1. Load and move data (SIMULATION)
            img_x, labels, target_params, target_coords, target_eos = to_device(batch,model.device)
            img_x = img_x.unsqueeze(1)


            optimizer.zero_grad()

            # 2. Forward Pass (Encoder + Initial LSTM states)
            pred_params, h0, c0, context_features, text_context = model(img_x, labels)

            # 3. Parameter Loss (mu, S, theta components)
            loss_params = criterion_params(pred_params, target_params)

            # 4. Sequence Decoding (Teacher Forcing uses target_coords as input)
            pred_coords, pred_eos_logits = decode_lstm_sequence(model, h0, c0, context_features,text_context, target_coords)

            # 5. Sequence Loss (Coordinate + EOS)
            loss_coord, loss_eos = criterion_sequence(pred_coords, pred_eos_logits, target_coords, target_eos)

            # 6. Combine Losses
            total_loss = (lambda_params * loss_params) + \
                        (lambda_seq_coord * loss_coord) + \
                        (lambda_seq_eos * loss_eos)

            # 7. Backward Pass and Step
            total_loss.backward()
            # Optional: Clip gradients to prevent exploding gradients common in RNNs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += total_loss.item(); running_p += loss_params.item(); running_c += loss_coord.item(); running_e += loss_eos.item()

            if (i) % report_rate == 0:
                if i>0: running_loss /= report_rate; running_p/=report_rate; running_c/=report_rate; running_e/=report_rate
                time_ = time.strftime("%H:%M:%S", time.localtime())
                log_ = f"{time_} :: Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss:.4f} [P: {running_p:.4f}, C: {running_c:.4f}, E: {running_e:.4f}]"
                print(log_); logging_str+=log_
                running_loss = 0.0; running_p = 0.0; running_c = 0.0; running_e = 0.0

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                # 1. Load and move data (SIMULATION)
                img_x, labels, target_params, target_coords, target_eos = to_device(batch,model.device)
                img_x = img_x.unsqueeze(1)


                pred_params, h0, c0, context_features, text_context = model(img_x, labels)
                pred_coords, pred_eos_logits = decode_lstm_sequence(model, h0, c0, context_features,text_context, target_coords)

                loss_params = criterion_params(pred_params, target_params)
                loss_coord, loss_eos = criterion_sequence(pred_coords, pred_eos_logits, target_coords, target_eos)

                total_loss = (lambda_params * loss_params) + \
                            (lambda_seq_coord * loss_coord) + \
                            (lambda_seq_eos * loss_eos)

                val_loss += total_loss.item()

        val_loss /= len(test_dataloader)
        time_ = time.strftime("%H:%M:%S", time.localtime())
        log_ = f"{time_} :: Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}"
        print(log_); logging_str+=log_
    if save_model:
        save_model_weights_and_args(model,f"{os.path.dirname(os.path.realpath(__file__))}/saved_models/{time.strftime('%m%d_%H%M%S', time.localtime())}_artist_model.pth",logs=logging_str)
    return model

def main(args=None):
    # subset_labels = ['apple', 'banana', 'bicycle', 'car', 'cat']
    subset_labels = ['apple', 'cat']
    subsample_dataset_ratio = 1.00
    train_test_split_ratio = .8
    batch_size = 8
    num_epochs = 8
    model_kwargs={'depth':5,'hidden_size':1024,'lang_hidden_size':128,}

    train_loader, test_loader, max_len = build_doodle_dataset(batch_size,subsample_dataset_ratio=subsample_dataset_ratio,
                                                             train_test_split_ratio=train_test_split_ratio,subset_labels=subset_labels)

    model = train_stroke_model(build_stroke_model(max_len,**model_kwargs),train_loader,test_loader,num_epochs=num_epochs)



if __name__ == '__main__':
    main()