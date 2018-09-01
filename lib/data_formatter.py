import os
import numpy as np
import pandas as pd

class DataFormatter:
    COLUMNS = [
        'Observed aval. hazard',
        'Max Temp Grad',
        'Max Hardness Grad',
        'Total Snow Depth',
        'Drift',
        'Snow Temp',
        'Foot Pen',
        'No Settle',
        'Insolation',
        'Rain at 900',
        'Air Temp',
        'Wind Dir',
        'Wind Speed',
        'Summit Air Temp',
        'Summit Wind Speed',
        'Summit Wind Dir',
        'Precip Code',
        'Crystals',
        'Wetness'
    ]
    LABEL = 'Observed aval. hazard'
    WIND_DIR = 'Wind Dir'
    SUMMIT_WIND_DIR = 'Summit Wind Dir'
    PRECIP = 'Precip Code'
    CRYSTAL = 'Crystals'

    def __init__(self, profiles=None):
        self.profiles = profiles

        x, y = [], []
        datasets = []

        for root, dirs, files in os.walk(self.profiles):
            for file_path in files:
                path = root + '/' + file_path
                dataset = pd.read_csv(path, index_col=False, usecols=self.COLUMNS, skipinitialspace=True)

                # apply numerical avalanche risk
                dataset[self.LABEL] = dataset[self.LABEL].apply(self._numerical_labels)
                # convert wind bearings to directions
                dataset[self.WIND_DIR] = dataset[self.WIND_DIR].apply(self._bearing_classification)
                dataset[self.SUMMIT_WIND_DIR] = dataset[self.SUMMIT_WIND_DIR].apply(self._bearing_classification)

                dataset = dataset.fillna(method='bfill')
                dataset = dataset.dropna()

                dataset = dataset.iloc[::-1]
                dataset['file_path'] = file_path
                datasets.append(dataset)

        datasets = pd.concat(datasets)
        datasets = pd.get_dummies(datasets, columns=[self.WIND_DIR,
            self.SUMMIT_WIND_DIR, self.PRECIP, self.CRYSTAL])

        for file_name, dataset in datasets.groupby('file_path'):
            dataset = dataset.drop(columns=['file_path'])

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

