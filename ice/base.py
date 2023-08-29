from abc import ABC, abstractmethod
import pandas as pd
import torch

class BaseModel(ABC):

    @abstractmethod
    def __init__(self, batch_size, lr, num_epochs, device, verbose):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: pd.Series):
        pass
    
    @abstractmethod
    def predict(self, sample: torch.FloatTensor):
        pass
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, target: pd.Series):
        pass
