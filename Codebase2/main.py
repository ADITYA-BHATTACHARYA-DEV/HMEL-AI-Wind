# File 4: main.py

from data_and_features import DataLoader, FeatureEngineer
from models import BiLSTM, PINN
from training_inference import Trainer, Inference
import torch
from torch.utils.data import DataLoader as TorchDL, TensorDataset


def prepare_tensors(df):
    X = torch.tensor(df[['dir_sin','dir_cos']].values, dtype=torch.float32)
    # reshape for sequence models: (batch, seq, features)
    X_seq = X.unsqueeze(0)  # batch size 1
    # dummy target (if none available)
    y = torch.zeros((X.shape[0],1), dtype=torch.float32)
    return X_seq, y


def main(filepath='wind.csv'):
    dl = DataLoader(filepath)
    df = dl.load()
    fe = FeatureEngineer()
    df = fe.add_circular_features(df)

    X_seq, y = prepare_tensors(df)

    model = BiLSTM(input_dim=2)
    trainer = Trainer(model)

    # Create a small DataLoader for demo
    dataset = TensorDataset(X_seq.squeeze(0), y)
    loader = TorchDL(dataset, batch_size=1)

    trainer.fit(loader, epochs=20)

    inf = Inference(model)
    pred = inf.predict(X_seq)
    print('Predicted:', pred)


if __name__ == '__main__':
    main()
