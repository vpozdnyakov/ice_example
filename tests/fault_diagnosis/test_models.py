import numpy as np
import pandas as pd
import pytest
from inspect import getmembers, isclass, isabstract
from ice.fault_diagnosis import models

models = [f[1] for f in getmembers(models, isclass) if not isabstract(f[1])]

class TestOnSyntheticData:
    def setup_class(self):
        df0 = pd.DataFrame({
            'sensor_0': np.sin(np.linspace(0, 20, 100)),
            'sensor_1': np.sin(np.linspace(0, 10, 100)),
            'sample': np.arange(100),
            'run_id': 0,
        })
        df1 = pd.DataFrame({
            'sensor_0': np.sin(np.linspace(0, 30, 100)),
            'sensor_1': np.sin(np.linspace(0, 15, 100)),
            'sample': np.arange(100),
            'run_id': 1,
        })
        self.df = pd.concat([df0, df1]).set_index(['run_id', 'sample'])
        self.target = pd.Series([0] * 100 + [1] * 100, index=self.df.index)

    @pytest.mark.parametrize("_model", models)
    def test_evaluate(self, _model):
        model = _model(window_size=10)
        model.fit(self.df, self.target)
        metrics = model.evaluate(self.df, self.target)
        assert metrics['accuracy'] >= 0
