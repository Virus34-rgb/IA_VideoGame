from torch import nn

class SelectionNetwork(nn.Module):
    def __init__(self,input_size=46, output_size=15):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.network(x)
        return logits