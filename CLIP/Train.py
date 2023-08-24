import torch
import torch.nn as nn
import clip
from DataLoader import get_data_loader
from Model import get_model_and_optimizer

batch_size = 200
excel_data = "/home/masai/PycharmProjects/CLIP-main/8.15.xlsx"
data_dir = "/home/masai/PycharmProjects/CLIP-main/image_data/001"
learning_rate = 0.0001
input_size = 1024
hidden_size = 512

cri = nn.CrossEntropyLoss()
device = "cuda:0"

# Load CLIP model
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# Get data loader
dataloader = get_data_loader(excel_data, data_dir, batch_size)

# Get model and optimizer
mlp, opt = get_model_and_optimizer(input_size, hidden_size, learning_rate)

for epoch in range(4000):
    total_loss = 0
    for sentence, image in dataloader:
        image = model.encode_image(image)
        sentence = model.encode_text(sentence)
        combined_features = torch.cat((sentence, image), dim=1).to(torch.float32)

        # Forward pass
        output = mlp(combined_features)
        output = output / output.norm(dim=-1, keepdim=True)
        logits = torch.matmul(output, output.T) * torch.exp(torch.tensor(0.1, device=device))

        # Compute loss and perform backpropagation
        labels = torch.arange(batch_size).to(device)
        loss = cri(logits, labels)

        # Backpropagation and optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


