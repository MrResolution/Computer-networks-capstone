import torch
import torch.nn as nn

class NetworkMalwareClassifier(nn.Module):
    """
    Hybrid 1D-CNN + BiLSTM Deep Learning Model for Network Malware Detection & Threat Classification.
    Combines spatial convolution across statistical flow metrics with bi-directional temporal LSTM processing.
    """
    def __init__(self, num_features=78, num_classes=8):
        super(NetworkMalwareClassifier, self).__init__()
        
        # 1D Convolutional Spatial Feature Extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Bi-directional LSTM Sequence Processor
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully-Connected Threat Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, num_features) -> unsqueeze to (batch_size, 1, num_features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        conv_out = self.conv_block(x)               # (batch_size, 128, num_features // 2)
        conv_out = conv_out.permute(0, 2, 1)        # (batch_size, seq_len, 128)
        
        lstm_out, _ = self.lstm(conv_out)           # (batch_size, seq_len, 128)
        last_timestep = lstm_out[:, -1, :]          # (batch_size, 128)
        
        logits = self.classifier(last_timestep)     # (batch_size, num_classes)
        return logits
