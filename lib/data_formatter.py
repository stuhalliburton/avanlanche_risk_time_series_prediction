import os
import numpy as np
import pandas as pd

class DataFormatter:
    COLUMNS = [
        'Observed aval. hazard',
        'Max Temp Grad',
        'Max Hardness Grad',
        # 'Total Snow Depth',
        'Drift',
        'Snow Temp',
        'Foot Pen',
        'No Settle',
        'Insolation',
        'Rain at 900',
        # 'Air Temp',
        # 'Summit Air Temp',
        # 'Summit Wind Speed',
        # 'Summit Wind Dir',
        # 'Precip Code',
        # 'Crystals'
    ]
    LABEL = 'Observed aval. hazard'
    TEMP_GRAD = 'Max Temp Grad'
    HARD_GRAD = 'Max Hardness Grad'
    INSOLATION = 'Insolation'
    SNOW_TEMP = 'Snow Temp'
    FOOT_PEN = 'Foot Pen'

    def create_dataset(self, look_back=1):
        x, y = [], []

        for root, dirs, files in os.walk('./profiles/southern-cairngorms/'):
            for file_path in files:
                path = root + file_path
                dataset = pd.read_csv(path, index_col=False, usecols=self.COLUMNS, skipinitialspace=True)
                dataset[self.LABEL] = dataset[self.LABEL].apply(self._numerical_labels)

                dataset[self.TEMP_GRAD] = dataset[self.TEMP_GRAD].fillna(method='bfill')
                dataset = dataset.dropna(subset=[self.TEMP_GRAD])

                dataset[self.HARD_GRAD] = dataset[self.HARD_GRAD].fillna(method='bfill')
                dataset = dataset.dropna(subset=[self.HARD_GRAD])

                dataset[self.INSOLATION] = dataset[self.INSOLATION].fillna(method='bfill')
                dataset = dataset.dropna(subset=[self.INSOLATION])

                dataset[self.SNOW_TEMP] = dataset[self.SNOW_TEMP].fillna(method='bfill')
                dataset = dataset.dropna(subset=[self.SNOW_TEMP])

                dataset[self.FOOT_PEN] = dataset[self.FOOT_PEN].fillna(method='bfill')
                dataset = dataset.dropna(subset=[self.FOOT_PEN])

                dataset = dataset.iloc[::-1]
                dataset = dataset.reset_index(drop=True)

                for index, row in dataset.iterrows():
                    if index < look_back:
                        continue
                    look_back_index = index - look_back
                    previous = dataset[look_back_index:index]
                    prediction = row[self.LABEL]
                    x.append(previous.values)
                    y.append(prediction)

        return np.array(x), np.array(y)

    def _numerical_labels(self, data):
        if data == 'Low':
            return 0
        if data == 'Moderate':
            return 1
        if data == 'Considerable -':
            return 2
        if data == 'Considerable +':
            return 3
        if data == 'High':
            return 4
        return 0

    def _bearing_classification(self, bearing):
        if 0 <= bearing <= 22.5:
            return 'n'
        if 337.6 <= bearing <= 360:
            return 'n'
        if 22.6 <= bearing <= 67.5:
            return 'ne'
        if 67.6 <= bearing <= 112.5:
            return 'e'
        if 112.6 <= bearing <= 157.5:
            return 'se'
        if 157.6 <= bearing <= 202.5:
            return 's'
        if 202.6 <= bearing <= 247.5:
            return 'sw'
        if 247.6 <= bearing <= 292.5:
            return 'w'
        if 292.6 <= bearing <= 337.5:
            return 'nw'
        return 'no wind'

