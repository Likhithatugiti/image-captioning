import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import json

class CaptionDataset(Dataset):
    def __init__(self, data_folder, transform, split='train'):
        self.split = split
        self.h = h5py.File(f"{data_folder}/caption_{split}.h5", 'r')
        self.imgs = list(self.h['images'].keys())
        self.word_map = json.load(open(f"{data_folder}/wordmap.json", 'r'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_id = self.imgs[idx]
        img = torch.FloatTensor(self.h['images'][img_id])  # Pre-extracted features
        caption = torch.LongTensor(self.h['captions'][img_id])
        return img, caption

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    caplens = torch.tensor([len(cap) for cap in caps])
    return imgs, caps, caplens
