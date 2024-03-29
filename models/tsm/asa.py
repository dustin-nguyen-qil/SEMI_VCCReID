import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .attention import SelfAttention

class ASA(nn.Module):
    def __init__(self,
                 rnn_size, 
                 input_size,
                 num_shape_params,
                 num_layers,
                 output_size,
                 feature_pool='concat',
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):
        super(ASA, self).__init__()
        self.input_size = input_size 
        self.num_shape_params = num_shape_params
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.layer1 = nn.Linear(in_features=self.input_size, out_features=self.num_shape_params)

        self.gru = nn.GRU(self.num_shape_params, self.rnn_size, num_layers=self.num_layers)

        linear_size = self.rnn_size if not self.feature_pool == 'concat' else self.rnn_size * 2

        if feature_pool == 'attention':
            self.attention = SelfAttention(
                attention_size=self.attention_size,
                layers=self.attention_layers,
                dropout=self.attention_dropout
            )
        
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, out_features=output_size))
        else:
            self.fc = nn.Linear(linear_size, out_features=output_size)

    # takes in framewise shape parameters sequence
    def forward(self, sequence):
        #sequence shape: batch_size, sequence length, input_size
        batch_size, seq_len, input_size = sequence.shape

        framewise_shape = self.layer1(sequence)
        outputs, state = self.gru(framewise_shape)

        if self.feature_pool == 'concat':
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batch_size, -1)
            max_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batch_size, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == 'attention':
            y, attentions = self.attention(outputs)
            output =  self.fc(y)
        else:
            output = self.fc(outputs[-1])
        
        videowise_shape = output
        
        return framewise_shape, videowise_shape