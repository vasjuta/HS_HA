import json
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class DataLoader:

    @staticmethod
    def read_csv(filename, labels_column):
        full_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(full_path, header=0, converters={labels_column: eval})
        return df
