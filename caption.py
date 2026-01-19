import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import EncoderDecoder
from datasets import WordMap
import json

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='path to image')
parser.add_argument('--model', type=str, required=True, help='path to model')
parser.add_argument('--word_map', type=str, default='data/wordmap.json')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(args.word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = EncoderDecoder(256, 512, len(word_map), 512).to(device)
model.load_state_dict(torch.load(args.model)['model'])
model.eval()

image = Image.open(args.img).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)
caption = model.caption_image(image_tensor, word_map)
caption_str = ' '.join([rev_word_map[idx.item()] for idx in caption])
print(caption_str)
