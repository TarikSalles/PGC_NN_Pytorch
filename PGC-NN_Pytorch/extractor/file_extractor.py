import json

import pandas as pd
import os
from tensorflow.keras.models import load_model
from scipy import sparse

class FileExtractor:

    def __init__(self):
        pass

    def read_csv(self, filename, dtypes_columns=None):

        if dtypes_columns is None:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, dtype=dtypes_columns, encoding='utf-8')

        return df.sample(frac=1, random_state=3)

    def read_multiples_csv(self, dir):

        files = [os.listdir(dir)[0]]

        concat_df = None
        for file in files:
            df = self.read_csv(dir+file)
            if concat_df is None:
                concat_df = df
            else:
                concat_df = pd.concat([concat_df, df], ignore_index=True)

        return concat_df


    def extract_ground_truth_from_csv(self):
        df = pd.read_csv(self.ground_truth_filename)

        return df

    def read_model(self, filename):

        return load_model(filename)

    def read_json(self, filename):

        with open(filename) as file:
            data = json.loads(file.read())

        return data

    def read_npz(self, filename):

        return sparse.load_npz(filename)