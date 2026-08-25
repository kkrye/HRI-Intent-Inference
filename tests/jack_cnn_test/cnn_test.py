# %% IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import time

from .trainer import build_doodle_dataset, build_stroke_model,train_stroke_model


# %% Build Stroke Model
model_depth = 3
input_img_size = 256
learning_rate = 1e-4
num_epochs = 3

# Loss Weights: Tune these carefully. Parameter loss often needs a higher weight.
lambda_seq_coord = 1.0
lambda_seq_eos = 0.25
lambda_params = 10.0
seq_loss_res = 64
seq_loss_sig = .025
size_lang_hidden = 32

# subset_labels = ['apple', 'banana', 'bicycle', 'car', 'cat']
subset_labels = ['apple', 'cat']
subsample_dataset_ratio = 0.1
train_test_split_ratio = .8
batch_size = 4

train_loader, test_loader, max_len = build_doodle_dataset(batch_size,subsample_dataset_ratio=subsample_dataset_ratio,
                                                            train_test_split_ratio=train_test_split_ratio,subset_labels=subset_labels)

model = train_stroke_model(build_stroke_model(max_len),train_loader,test_loader,num_epochs=num_epochs,lambda_params=lambda_params,lambda_seq_coord=lambda_seq_coord,lambda_seq_eos=lambda_seq_eos,learning_rate=learning_rate)

# %% Test Inference
# force_label = 'apple'
force_label = 'cat'
test_idx = 955
for i,(img_x, labels, target_params, target_coords, target_eos) in enumerate(test_dataloader):
    if i>=test_idx:
        break
img_x = img_x.unsqueeze(1).to(model.device)
final_stroke, p, s, eos = run_inference_walkthrough(model,img_x[:1],labels[:1])
true_stroke = transform_stroke(target_params[0].cpu().detach().numpy(), target_coords[0].cpu().detach().numpy(), target_eos[0].cpu().detach().numpy())
p_img = img_x[0].cpu().detach().numpy()[0]
plt.imshow(p_img,cmap='gray');plt.plot(final_stroke[:,0],final_stroke[:,1]);plt.plot(true_stroke[:,0],true_stroke[:,1])
plt.title(f"Truth:{labels[0]}, Forced:{force_label} Testing:{test_idx}")
# %%
