import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=512, h_RNN_layers=2, h_RNN=512, h_FC_dim=256, drop_p=0.5, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.rgb_LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True, # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
            dropout = 0.5,
        )

        """
        self.FLOW = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True, # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
            dropout = 0.5,
        )
        """

        self.fc1 = nn.Linear(self.h_RNN*2, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        #self.atten_lstm = nn.Linear(self.h_RNN*2, 16)

    def forward(self, x_RNN):
        self.rgb_LSTM.flatten_parameters()
        RNN_rgb_out, (h_n, h_c) = self.rgb_LSTM(x_RNN, None)
        #self.FLOW.flatten_parameters()
        #RNN_flow_out, (h_n, h_c) = self.FLOW(x_RNN[:, 1:, :] - x_RNN[:,:15,:], None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # Atten
        #atten = self.atten_lstm(torch.cat((h_c[-1,:], h_c[-2,:]),1))  # [batch, time_step]
        #atten = torch.nn.functional.softmax(atten)
        #atten = torch.bmm(atten.unsqueeze(1), RNN_out).squeeze(1) # batch, out_size
        # FC layers
        x = self.fc1(RNN_rgb_out[:,-1,:])  # choose RNN_out at the last time step
        #x = self.fc1(torch.cat((RNN_rgb_out[:, -1, :], RNN_flow_out[:, -1, :]), 1))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
