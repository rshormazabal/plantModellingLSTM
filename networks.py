import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0, bidirectional=True):
        super(LSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.layer_dim,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=self.bidirectional
                            )

        # Define the output layer, dimensions depend if bidirectional or not
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden states, dimensions depend on if bidirectional or not
        if self.bidirectional:
            h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
            c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print('Element 0 Min: {}, Max: {}, Mean: {}'.format(self.lstm.all_weights[0][0].min(),
        #                                                    self.lstm.all_weights[0][0].max(),
        #                                                    self.lstm.all_weights[0][0].mean()))

        out = self.fc(out[:, -1, :])
        return out