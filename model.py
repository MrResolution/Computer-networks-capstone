import torch
import torch.nn as nn

class NetworkMalwareClassifier(nn.Module):
    """
    Hybrid 1D-CNN + BiLSTM Deep Learning Model for Network Malware Detection & Threat Classification.
    Combines spatial convolution across statistical flow metrics with bi-directional temporal LSTM processing.
    
    The architecture uses two parallel paths:
    1. CNN path: captures local feature patterns via 1D convolutions
    2. LSTM path: captures sequential dependencies across the feature vector
    Both paths are concatenated and fed to the classification head.
    """
    def __init__(self, num_features=78, num_classes=8):
        super(NetworkMalwareClassifier, self).__init__()
        
        # 1D Convolutional Spatial Feature Extractor (padded to preserve length)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pool → (batch, 128, 1)
        )
        
        # Bi-directional LSTM Sequence Processor
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Direct fully-connected path for tabular features
        self.fc_direct = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fully-Connected Threat Classification Head
        # Input = CNN(128) + LSTM(128) + Direct(64) = 320
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, num_features)
        if len(x.shape) == 2:
            x_2d = x  # Keep original 2D for direct FC path
            x_3d = x.unsqueeze(1)  # (batch, 1, num_features) for Conv1d
        else:
            x_2d = x.squeeze(1)
            x_3d = x
        
        # CNN path
        conv_out = self.conv_block(x_3d)        # (batch, 128, 1)
        conv_out = conv_out.squeeze(-1)          # (batch, 128)
        
        # LSTM path
        lstm_in = x_2d.unsqueeze(-1)             # (batch, num_features, 1)
        lstm_out, _ = self.lstm(lstm_in)          # (batch, num_features, 128)
        lstm_last = lstm_out[:, -1, :]            # (batch, 128) — last timestep
        
        # Direct FC path
        fc_out = self.fc_direct(x_2d)             # (batch, 64)
        
        # Concatenate all paths
        combined = torch.cat([conv_out, lstm_last, fc_out], dim=1)  # (batch, 320)
        
        logits = self.classifier(combined)        # (batch, num_classes)
        return logits
