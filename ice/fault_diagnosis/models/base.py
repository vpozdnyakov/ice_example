from abc import ABC, abstractmethod
import pandas as pd
from tqdm.auto import trange, tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ice.fault_diagnosis.data import SlidingWindowDataset
from ice.base import BaseModel
from ice.fault_diagnosis.metrics import accuracy


class BaseFaultDiagnosis(BaseModel, ABC):

    @abstractmethod
    def __init__(self, window_size, batch_size, lr, num_epochs, device, verbose):
        super().__init__(batch_size, lr, num_epochs, device, verbose)
        self.window_size = window_size
        self.model = None
        self.loss_fn = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: pd.Series):
        assert len(df) >= self.window_size
        num_classes = len(set(target))
        weight = torch.ones(num_classes, device=self.device) * 0.5
        weight[1:] /= num_classes - 1
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def predict(self, sample: torch.FloatTensor):
        self.model.eval()
        self.model.to(self.device)
        sample = sample.to(self.device)
        with torch.no_grad():
            logits = self.model(sample)
        return logits.argmax(axis=1).cpu()
    
    def evaluate(self, df: pd.DataFrame, target: pd.Series):
        self.model.eval()
        self.model.to(self.device)

        dataset = SlidingWindowDataset(df, target, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        target, pred = [], []
        for sample, _target in tqdm(
            self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)
        ):
            sample = sample.to(self.device)
            target.append(_target)
            with torch.no_grad():
                pred.append(self.predict(sample))
        target = torch.concat(target).numpy()
        pred = torch.concat(pred).numpy()
        metrics = {
            'accuracy': accuracy(pred, target)
        }
        return metrics

    def train_nn(self, df: pd.DataFrame, target: pd.Series):
        self.model.train()
        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        dataset = SlidingWindowDataset(df, target, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for e in trange(self.num_epochs, desc='Epochs ...', disable=(not self.verbose)):
            for sample, target in tqdm(self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)):
                sample = sample.to(self.device)
                target = target.to(self.device)
                logits = self.model(sample)
                loss = self.loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f'Epoch {e+1}, Loss: {loss.item():.4f}')
