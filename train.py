import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import CaptionDataset, collate_fn
from models import EncoderDecoder
from pycocoevalcap.bleu.bleu import Bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
embed_size = 256
attention_dim = 512
decoder_dim = 512
learning_rate = 4e-4
data_folder = 'data/'
word_map_path = 'data/wordmap.json'
num_epochs = 20

train_loader = DataLoader(CaptionDataset(data_folder, None, 'train'),
                          batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader =  ValLoader(CaptionDataset(data_folder, None, 'train'),
                          batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = EncoderDecoder(embed_size, decoder_dim, len(word_map), attention_dim).to(device)
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        outputs = model(imgs, caps[:, 1:], caplens - 1)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), caps[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
    scheduler.step()
    # Val BLEU
    recent_bleu4 = evaluate(model, val_loader, word_map)
    torch.save({'epoch': epoch, 'model': model.state_dict()}, f"models/BEST.pth.tar")
