import torch
import numpy as np
import pandas as pd
import json

import cv2


class PartialImageDatasetPreLoad(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.partial_img_data = []
        self.partial_img_label = []
        self.partial_img_next = []
        blank_img = np.full((256, 256),255, dtype = np.uint8)
        for idx, drawing in enumerate(image_df['drawing']):
            img = blank_img.copy()
            strokes = json.loads(drawing)
            for stroke_num in range(len(strokes)-1):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
                self.partial_img_data.append(img.copy())
                self.partial_img_label.append(image_df['word'][idx])
                just_next_stroke = self.add_stroke_to_image(blank_img, strokes[stroke_num+1])
                self.partial_img_next.append(just_next_stroke)
        self.partial_img_data = np.array(self.partial_img_data)
        self.partial_img_label = np.array(self.partial_img_label)
        self.partial_img_next = np.array(self.partial_img_next)
        assert self.partial_img_data.shape[0] == self.partial_img_label.shape[0] == self.partial_img_next.shape[0]

    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.partial_img_data.shape[0]

    def __getitem__(self, idx):
        return self.partial_img_data[idx], self.partial_img_label[idx], self.partial_img_next[idx]

class PartialImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        super().__init__()
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.cum_sum_strokes = []
        for drawing in (image_df['drawing']):
            strokes = json.loads(drawing)
            self.cum_sum_strokes.append(len(strokes)-1)
        self.cum_sum_strokes = np.cumsum(self.cum_sum_strokes)

    def get_subidx(self, idx):
        sample_idx = np.searchsorted(self.cum_sum_strokes, idx, side='right')
        if sample_idx == 0:
            stroke_idx = idx
        else:
            stroke_idx = idx - self.cum_sum_strokes[sample_idx-1]
        return sample_idx, stroke_idx+1


    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.cum_sum_strokes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            sample_idx, final_stroke_idx = self.get_subidx(idx)
            blank_img = np.full((256, 256),255, dtype = np.uint8)
            img = blank_img.copy()
            strokes = json.loads(self.raw_data['drawing'][sample_idx])
            for stroke_num in range(final_stroke_idx):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
            partial_img_data = img.copy()
            partial_img_label = self.raw_data['word'][sample_idx]
            just_next_stroke = self.add_stroke_to_image(blank_img, strokes[final_stroke_idx])

            return partial_img_data, partial_img_label, just_next_stroke

def to_device(data, device):
    """
    Recursively moves all tensors in a data structure (list, tuple, dict) to a specific device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data # Return non-tensor data as is

def raw_stroke_to_normed(stroke:np.ndarray,max_stroke_len:int, image_size:int=256):
    stroke_swap = np.swapaxes(stroke,1,0).astype(dtype=np.float32)
    mu_arr = stroke_swap.mean(axis=0)
    dxdy = stroke_swap[-1] - stroke_swap[0]
    # s = np.sqrt(np.square(stroke_swap.max(axis=0) - stroke_swap.min(axis=0)).sum())
    s = (stroke_swap.max(axis=0) - stroke_swap.min(axis=0)).max()
    s = np.maximum(s, 1e-6)
    r = np.sqrt(dxdy[0]**2 + dxdy[1]**2)
    r = np.maximum(r, 1e-6)
    cos_theta = dxdy[0] / r
    sin_theta = dxdy[1] / r
    normed_arr = np.zeros_like(stroke_swap)
    normed_arr[:,0] = ((stroke_swap[:,0] - mu_arr[0]) * cos_theta + (stroke_swap[:,1] - mu_arr[1]) * sin_theta) / s
    normed_arr[:,1] = ((stroke_swap[:,0] - mu_arr[0]) * sin_theta + (stroke_swap[:,1] - mu_arr[1]) * cos_theta) / s
    pad_len = max_stroke_len - stroke_swap.shape[0]
    padded_stroke = np.pad(normed_arr,((0,pad_len),(0,0)),mode='edge')

    eos_arr = np.zeros((max_stroke_len,),dtype=padded_stroke.dtype)
    eos_arr[stroke.shape[1]-1] += 1

    params_arr = np.array([mu_arr[0]/image_size, mu_arr[1]/image_size, s/image_size, cos_theta, sin_theta])

    return params_arr, padded_stroke, eos_arr

def transform_stroke(params_arr, padded_stroke, eos_arr,image_size=256):
    normed_stroke = padded_stroke[:int(np.where(eos_arr==1)[0][0])+1].copy()
    normed_stroke *= (params_arr[2]*image_size)
    stroke = np.zeros_like(normed_stroke)
    # stroke[:,0] = params_arr[0] + normed_stroke[:,0] * params_arr[3] - normed_stroke[:,0] * params_arr[4]
    # stroke[:,1] = params_arr[1] - normed_stroke[:,0] * params_arr[4] + normed_stroke[:,0] * params_arr[3]

    stroke[:,0] = params_arr[0]*image_size + (params_arr[3] * normed_stroke[:,0] - params_arr[4] * normed_stroke[:,1]) / (params_arr[3]**2 - params_arr[4]**2)
    stroke[:,1] = params_arr[1]*image_size + (params_arr[4] * normed_stroke[:,0] - params_arr[3] * normed_stroke[:,1]) / (params_arr[4]**2 - params_arr[3]**2)
    return stroke

class PartialImageStrokeDataset(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame, max_stroke_len:int, image_size=256):
        super().__init__()
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.stroke_len = max_stroke_len
        self.cum_sum_strokes = []
        self.image_size = image_size
        for drawing in (image_df['drawing']):
            strokes = json.loads(drawing)
            self.cum_sum_strokes.append(len(strokes)-1)
        self.cum_sum_strokes = np.cumsum(self.cum_sum_strokes)
        print(f"Strokes forced to length of {self.stroke_len}")

    def get_subidx(self, idx):
        sample_idx = np.searchsorted(self.cum_sum_strokes, idx, side='right')
        if sample_idx == 0:
            stroke_idx = idx
        else:
            stroke_idx = idx - self.cum_sum_strokes[sample_idx-1]
        return sample_idx, stroke_idx+1


    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.cum_sum_strokes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            sample_idx, final_stroke_idx = self.get_subidx(idx)
            blank_img = np.full((self.image_size, self.image_size),255, dtype = np.uint8)
            img = blank_img.copy()
            strokes = json.loads(self.raw_data['drawing'][sample_idx])
            for stroke_num in range(final_stroke_idx):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
            partial_img_data = img.copy().astype(np.float32)
            partial_img_label = self.raw_data['word'][sample_idx]
            final_stroke = np.array(strokes[final_stroke_idx]).astype(np.float32)
            stroke_params, stroke_normed, stroke_eos = raw_stroke_to_normed(final_stroke,max_stroke_len=self.stroke_len,image_size=self.image_size)

            return partial_img_data, partial_img_label, stroke_params.astype(np.float32), stroke_normed, stroke_eos


def build_doodle_dataset(batch_size=4,subsample_dataset_ratio=.1,train_test_split_ratio=.8,subset_labels=None,full_img_loader=False,split_seed=42):
    dataset_path = "data/master_doodle_dataframe.csv"
    full_raw_data = pd.read_csv(dataset_path)
    full_data = full_raw_data.drop(columns=["countrycode", "recognized", "key_id", "image_path"])
    if subset_labels is None:
        subset_labels = full_data['word'].unique().tolist()
    data = full_data[full_data['word'].isin(subset_labels)].reset_index(drop=True)
    data = data.sample(frac=subsample_dataset_ratio, random_state=split_seed).reset_index(drop=True)
    max_stroke_len_arr = np.array([max([len(s[0]) for s in json.loads(d)]) for d in data['drawing']])
    stroke_len = max_stroke_len_arr.max()
    train_df = data.sample(frac=train_test_split_ratio, random_state=split_seed)
    test_df = data.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    if full_img_loader:
        train_set = PartialImageDataset(train_df)
        test_set = PartialImageDataset(test_df)
    else:
        train_set = PartialImageStrokeDataset(train_df,max_stroke_len=stroke_len)
        test_set = PartialImageStrokeDataset(test_df,max_stroke_len=stroke_len)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Using subset labels: {subset_labels} and only {subsample_dataset_ratio*100}% of full dataset")
    print(f"Train-Test split is: {train_test_split_ratio}")
    return train_dataloader, test_dataloader, stroke_len