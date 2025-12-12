# File 3: training_inference.py

import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, lr=1e-3, device=None):
        self.model = model
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            self.opt.zero_grad()
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, dataloader, epochs=50, val_loader=None):
        for ep in range(epochs):
            train_loss = self.train_epoch(dataloader)
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {ep} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {ep} train_loss={train_loss:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                total_loss += self.loss_fn(pred, yb).item()
        return total_loss / len(dataloader)


class Inference:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, x_tensor):
        self.model.eval()
        x = x_tensor.to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()
