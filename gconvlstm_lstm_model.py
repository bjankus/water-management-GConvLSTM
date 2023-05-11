"""# **Model**"""
import json
import shutil
import torch
from torch.nn import LSTM
from torch_geometric_temporal.nn.recurrent import GConvLSTM


class LSTMSequenceModel(torch.nn.Module):
    """Custom model with one target"""

    def __init__(
            self,
            dropout: float = 0,
            hidden_size: int = 32,
            k: int = 1,
            model_name: str = 'Model',
            node_features: int = 4,
            target_len: int = 7,
            data_params: str = 'data_dict.json',
            **kwargs
    ):
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.k = k
        self.node_features = node_features
        self.model_name = model_name
        self.target_len = target_len
        self.data_params = data_params

        with open(self.data_params, 'r') as openfile:
            # Reading from json file
            dicti = json.load(openfile)

        self.idx_Szeged = dicti["Szeged"]["column_idx"]

        super().__init__(**kwargs)

        # save the hyperparameters
        # Data to be written
        parameters = {
            "dropout": self.dropout,
            "hidden_size": self.hidden_size,
            "k": self.k,
            "model_name": self.model_name,
            "node_features": self.node_features,
            "target_len": self.target_len,
            "data_params": self.data_params

        }

        # Serializing json
        json_object = json.dumps(parameters, indent=7)

        # Writing to parameters.json
        with open(str(self.model_name) + ".json", "w") as outfile:
            outfile.write(json_object)
            

        #####################
        # ENCODER LSTM - number of inputs is n_features
        self.lstm_encode = GConvLSTM(
            in_channels=self.node_features,
            out_channels=self.hidden_size,
            K=self.k  # Chebyshev filter size
        )
        # return H: Hidden state matrix (PyTorch Float Tensor) for all nodes.
        #        C: Cell state matrix (PyTorch Float Tensor) for all nodes.

        # DECODER LSTM - number of inputs is 1
        self.lstm_decode = LSTM(
            hidden_size=self.hidden_size,
            input_size=1,
            num_layers=1,  # Number of recurrent layers
            dropout=self.dropout
        )

        # OUTPUT TRANSFORMERS
        self.output_layer = torch.nn.Linear(in_features=self.hidden_size,
                                            out_features=1)

    def encode(self,
               x: torch.Tensor,
               edge_index: torch.Tensor,
               edge_weight: torch.Tensor,
               h: torch.Tensor,
               c: torch.Tensor
               ):
        hidden, cell = self.lstm_encode(X=x,
                                        edge_index=edge_index,
                                        edge_weight=edge_weight,
                                        H=h,
                                        C=c)

        return hidden, cell

    def decode(self,
               x: torch.Tensor,
               hidden_state: torch.Tensor):
        # SELECT FIRST INPUT
        # LAST ENCODER TARGET CAN BE USED
        # the present is the input[0] (from fig)
        input_vector = x[self.idx_Szeged, -1:].unsqueeze(0)

        # input_vector is torch.Size([1, 1])
        # dim0: time (I use the last timeindex)

        # INITIALIZE OUTPUTS [decoder_lengths, 1, 1]
        target_len = self.target_len  # how many days forecast
        outputs = torch.zeros(target_len, 1, 1)

        # PREDICT RECURSIVELY
        for t in range(target_len):
            # DECODER output [1, hidden_size]
            decoder_output, hidden_state = self.lstm_decode(input_vector, hidden_state)

            # TRANSFORM TO OUTPUT FORMAT [1, 1]
            decoder_output = self.output_layer(decoder_output)
            # COLLECT DECODER OUTPUTS
            outputs[t] = decoder_output

            # SELECT NEXT INPUT FOR THE NEXT STEP OF DECODER
            input_vector = decoder_output

        # RESHAPE OUTPUTS TO FORCASTING FORMAT
        # OUTPUTS [decoder_lengths, 1, 1] -> OUTPUTS [decoder_lengths, 1]
        prediction = torch.squeeze(outputs, 2)

        return prediction

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                h: torch.Tensor,
                c: torch.Tensor
                ) -> torch.Tensor:
        # ENCODE
        hidden, cell = self.encode(x=x,
                                   edge_index=edge_index,
                                   edge_weight=edge_weight,
                                   h=h,
                                   c=c)  # encode to hidden state

        # I need from H, C only what belongs to Szeged
        hidden_Szeged = hidden[self.idx_Szeged: self.idx_Szeged + 1, :]
        cell_Szeged = cell[self.idx_Szeged: self.idx_Szeged + 1, :]

        encoder_hidden = (hidden_Szeged, cell_Szeged)

        # DECODE
        prediction = self.decode(x=x,
                                 hidden_state=encoder_hidden)

        return prediction, self.idx_Szeged