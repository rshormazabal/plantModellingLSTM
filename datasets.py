import random

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class plantData(Dataset):
    # TODO: REWRITE TO DELIVER SEQUENCE, ADD ARGUMENT FOR TIME
    # TODO: JUST PICKLE THE LIST WITH THE VALID SEQUENCES
    # TODO: GET TIMESTAMP TOO
    def __init__(self, residence_time,  out_seq_dim=1,  target='IV', data_amount=1, shuffle=False):
        """

        :param out_seq_dim: length of predicted sequence
        :param residence_time: residence time in minutes
        :param target: target parameter to predict
        :param data_amount: fraction of data to use
        """
        super(plantData, self).__init__()

        # standarizer
        self.standarizer = StandardScaler()

        # load data and convert time
        self.data = pd.read_csv('./data/all_data_raw.csv')
        self.data.columns = ['Time'] + list(self.data.columns)[1:]
        self.data['Time'] = pd.to_datetime(self.data['Time'])

        # Creating input vector
        # (input_size, seq_length, feature_dim)
        self.residence_time = residence_time
        self.seq_dim = int(residence_time / 5)
        self.feature_dim = len(self.data.columns[5:])
        self.out_seq_dim = out_seq_dim
        self.target = target

        self.valid_sequences = []
        # iter from the first target with "seq_dim" steps before
        for i in range(self.seq_dim, self.data.shape[0]):
            # check if is number
            # TODO: Add checked to see if the particular value if suddently out of range(noise).
            if isinstance(self.data.loc[i, self.target], np.float64):
                null_sum = self.data.loc[i-self.seq_dim:i].isnull().sum(axis=0).sum()
                time_difference = (self.data.Time[i] - self.data.Time[i - self.seq_dim]).seconds
                # check if there any "NaN" value in the previous rows and
                # check ammount of time between first and last record (residence_time)
                if (null_sum == 0) and (time_difference == residence_time*60):
                    self.valid_sequences.append(i)

            if i % 1000 == 0:
                print("{} sequences verified".format(i))

        # choose a subset of data
        # TODO: CHECK data_amount parameter, originally to choose less data.
        if shuffle:
            self.valid_sequences = random.choices(self.valid_sequences, k=int(self.data.shape[0]*data_amount))

        # standarize
        self.standarizer.fit(self.data.iloc[:, 5:].to_numpy())

    def __getitem__(self, index):
        real_index = self.valid_sequences[index]

        X = self.data.iloc[real_index - self.seq_dim: real_index, 5:].to_numpy()
        label = self.data.loc[real_index: real_index + self.out_seq_dim - 1, self.target].to_numpy()

        features = self.standarizer.transform(X)

        return features, label

    def get_timestamp(self, index):
        """

        :param index:
        :return: get timestamp from original dataset
        """
        real_index = self.valid_sequences[index]
        return self.data.loc[real_index, 'Time'].to_pydatetime()

    def __len__(self):
        return len(self.valid_sequences)


class plantDataSeqToSeq(Dataset):
    # TODO: REWRITE TO DELIVER SEQUENCE, ADD ARGUMENT FOR TIME
    # TODO: JUST PICKLE THE LIST WITH THE VALID SEQUENCES
    # TODO: GET TIMESTAMP TOO
    def __init__(self, residence_time,  out_seq_dim=1,  target='IV', data_amount=1, shuffle=False):
        """

        :param out_seq_dim: length of predicted sequence
        :param residence_time: residence time in minutes
        :param target: target parameter to predict
        :param data_amount: fraction of data to use
        """
        super(plantDataSeqToSeq, self).__init__()

        # standarizer
        self.standarizer = StandardScaler()

        # load data and convert time
        self.data = pd.read_csv('./data/all_data_raw.csv')
        self.data.columns = ['Time'] + list(self.data.columns)[1:]
        self.data['Time'] = pd.to_datetime(self.data['Time'])

        # Creating input vector
        # (input_size, seq_length, feature_dim)
        self.residence_time = residence_time
        self.seq_dim = int(residence_time / 5)
        self.feature_dim = len(self.data.columns[5:])
        self.out_seq_dim = out_seq_dim
        self.target = target

        self.valid_sequences = []
        # iter from the first target with "seq_dim" steps before
        for i in range(self.seq_dim, self.data.shape[0]):
            # check if is number
            # TODO: Add checked to see if the particular value if suddently out of range(noise).
            if isinstance(self.data.loc[i, self.target], np.float64):
                null_sum = self.data.loc[i-self.seq_dim:i].isnull().sum(axis=0).sum()
                time_difference = (self.data.Time[i] - self.data.Time[i - self.seq_dim]).seconds
                # check if there any "NaN" value in the previous rows and
                # check ammount of time between first and last record (residence_time)
                if (null_sum == 0) & (time_difference == residence_time*60):
                    self.valid_sequences.append(i)

            if i % 1000 == 0:
                print("{} sequences verified".format(i))

        # choose a subset of data
        # TODO: CHECK data_amount parameter, originally to choose less data.
        if shuffle:
            self.valid_sequences = random.choices(self.valid_sequences, k=int(self.data.shape[0]*data_amount))

        # standarize
        self.standarizer.fit(self.data.iloc[:, 5:].to_numpy())

    def __getitem__(self, index):
        real_index = self.valid_sequences[index]

        X = self.data.iloc[real_index - self.seq_dim: real_index, 5:].to_numpy()
        features = self.standarizer.transform(X)

        # concatenate label with all the features in the current time step
        # (the one we are predicting at, example: +5min)
        # remember: iloc takes +1, loc does not.
        label_x = self.data.iloc[real_index: real_index + self.out_seq_dim, 5:].to_numpy()
        label_x = self.standarizer.transform(label_x)
        label_y = self.data.loc[real_index: real_index + self.out_seq_dim - 1, self.target].to_numpy()

        label = np.append(label_x, label_y)

        return features, label

    def get_timestamp(self, index):
        """
        :param index:
        :return: get timestamp from original dataset
        """
        real_index = self.valid_sequences[index]
        return self.data.loc[real_index, 'Time'].to_pydatetime()

    def __len__(self):
        return len(self.valid_sequences)