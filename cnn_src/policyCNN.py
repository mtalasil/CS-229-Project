import torch.nn as nn

#TODO
# add dropout
# check distribution of action labels in training and val set
class PolicyCNN(nn.Module):
    def __init__(self, in_channels:int, num_actions:int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 8x8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 128 x 1 x 1
            nn.Flatten(),             # 128
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (C,64,64) -> (32,64,64)
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (64,64,64)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),                                       # (64,32,32)
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),          # (128,32,32)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),                                       # (128,16,16)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),                                          # 128 * 16 * 16
        #     nn.Linear(128 * 16 * 16, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_actions)                            # logits for each action
        # )
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (C,64,64) -> (32,64,64)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
            
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (64,64,64)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),                                       # (64,32,32)

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),          # (128,32,32)
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),                                       # (128,16,16)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),                                          # 128 * 16 * 16
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     #nn.Dropout(p=0.3),
        #     nn.Linear(64, num_actions)                            # logits for each action
        # )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x