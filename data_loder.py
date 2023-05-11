
"""# **Data loading**
"""

import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class DatasetLoader(object):

    def __init__(self,
                 data: str = 'data.csv',
                 data_params: str = 'data_dict.json',
                 data_slice: tuple = (0, 501)):
        self.data = data
        self.data_params = data_params
        self.data_slice = data_slice

        self._read_data()

    def _read_data(self):
        new_data = np.loadtxt(fname=self.data,
                              usecols=(np.r_[1:13]),
                              # np.r_ generates an array of indices
                              delimiter=",")

        # The 0 row contains the station codes.
        # value = new_data[1:,:]
        # less data
        start, end = self.data_slice
        values = new_data[start:end, :]

        self.values = values

        with open(self.data_params, 'r') as openfile:
            # Reading from json file
            dict_data = json.load(openfile)

        # standardization
        std = np.array(list(d['std'] for d in dict_data.values() if d))
        mean = np.array(list(d['mean'] for d in dict_data.values() if d))

        std_data = (values - mean) / std

        #######################################x

        # column serial number for the city in the dataset
        d = {k: v["column_idx"] for k, v in dict_data.items()}

        self._dataset = {
            "edges": [[d["Szeged"], d["Zenta"]],
                      [d["Vásárosnemény"], d["Tokaj"]],
                      [d["Tokaj"], d["Tiszapalkonya"]],
                      [d["Tiszadorogma"], d["Csongrád"]],
                      [d["Mindszent"], d["Algyő"]],
                      [d["Algyő"], d["Szeged"]],
                      [d["Gyoma"], d["Mindszent"]],
                      [d["Makó"], d["Szeged"]],
                      [d["Tiszapalkonya"], d["Tiszadorogma"]],
                      [d["Békés"], d["Gyoma"]],
                      [d["Csongrád"], d["Mindszent"]]
                      ],
            "node_ids": d,
            "FX": std_data
        }

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i: i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags - self.target + 1)
        ]
        self.targets = [
            stacked_target[i + self.lags : i + self.lags + self.target, :].T
            for i in range(stacked_target.shape[0] - self.lags - self.target + 1)
        ]

    def get_dataset(self, lags: int=4, target: int=7) -> StaticGraphTemporalSignal:
        self.lags = lags
        self.target = target
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
