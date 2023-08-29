from torch import nn
from ice.fault_diagnosis.models.base import BaseFaultDiagnosis
import pandas as pd


class MLP(BaseFaultDiagnosis):
    def __init__(
            self, 
            window_size: int, 
            hidden_dim=256,
            batch_size=128,
            lr=0.001,
            num_epochs=10,
            device='cpu',
            verbose=False,
        ):
        super().__init__(
            window_size, batch_size, lr, num_epochs, device, verbose,
        )
        self.hidden_dim = hidden_dim

    def fit(self, df: pd.DataFrame, target: pd.Series):
        super().fit(df, target)
        num_sensors = df.shape[1]
        num_classes = len(set(target))
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        )
        super().train_nn(df, target)
