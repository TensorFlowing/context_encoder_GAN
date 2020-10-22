import torch
from torch import nn, optim


class CEGenerator(nn.Module): # context encoder
    def __init__(self, channels=3): # channels=input color channels
        super(CEGenerator, self).__init__() # super() calls the __init__() of the parent class,
        # define two layer templates
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(num_features=out_feat, eps=0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(num_features=out_feat, eps=0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            *upsample(64, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class CEDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(CEDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters)) #???
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    from utils.flops_estimation import add_flops_counting_methods

    G_net = CEGenerator()
    D_net = CEDiscriminator()
    print(G_net)
    print(D_net)

    fcn = add_flops_counting_methods(G_net)
    fcn = fcn.train()
    fcn.start_flops_count()

    img = torch.rand(1, 3, 256, 256)
    y = fcn(img)
    n = fcn.compute_average_flops_cost() / 1e9
  
    print(y.size())
    print(n)
