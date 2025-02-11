from torch import nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)