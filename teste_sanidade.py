import torch
from models.lalic_swhdc import LALIC

net = LALIC(use_swhdc=True, swhdc_dilations=(1, 2, 3)).cuda()
x = torch.randn(1, 3, 256, 512).cuda()
y = net.g_a(x)
print(y.shape)