import torch


class SqueezeAndExcitation(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1_1 = torch.nn.Linear(64, 64)
        self.fc1_2 = torch.nn.Linear(1024, 512)
        self.fc1_3 = torch.nn.Linear(48, 32)
        self.layernorm1_1 = torch.nn.LayerNorm(64)
        self.layernorm1_2 = torch.nn.LayerNorm(512)
        self.layernorm1_3 = torch.nn.LayerNorm(32)

        self.fc2_1 = torch.nn.Linear(64, 64)
        self.fc2_2 = torch.nn.Linear(512, 128)
        self.fc2_3 = torch.nn.Linear(32, 64)
        self.layernorm2_1 = torch.nn.LayerNorm(64)
        self.layernorm2_2 = torch.nn.LayerNorm(128)
        self.layernorm2_3 = torch.nn.LayerNorm(64)

        self.fc3 = torch.nn.Linear(256, 256)
        # self.se3 = SqueezeAndExcitation(256)
        self.se3 = torch.nn.Identity()
        self.leakyrelu1 = torch.nn.LeakyReLU(0.2)

        self.fc4 = torch.nn.Linear(256, 256)
        # self.se4 = SqueezeAndExcitation(128)
        self.se4 = torch.nn.Identity()
        self.leakyrelu2 = torch.nn.LeakyReLU(0.2)

        self.fc5 = torch.nn.Linear(256, 256)
        # self.se5 = SqueezeAndExcitation(64)
        self.se5 = torch.nn.Identity()
        self.leakyrelu3 = torch.nn.LeakyReLU(0.2)

        self.fc6 = torch.nn.Linear(256, 64)
        self.fc7 = torch.nn.Linear(64, 2)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        emb, gse, sub = x
        emb = self.layernorm1_1(self.fc1_1(emb))
        sub = self.layernorm1_2(self.fc1_2(sub))
        gse = self.layernorm1_3(self.fc1_3(gse))

        emb = self.layernorm2_1(self.fc2_1(emb))
        sub = self.layernorm2_2(self.fc2_2(sub))
        gse = self.layernorm2_3(self.fc2_3(gse))

        x = torch.cat([emb, gse, sub], dim=1)
        x = self.leakyrelu1(self.fc3(x)) + self.se3(x)
        x = self.leakyrelu2(self.fc4(x)) + self.se4(x)
        x = self.leakyrelu3(self.fc5(x)) + self.se5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        return self.sigmoid(x)