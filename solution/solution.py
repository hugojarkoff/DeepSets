class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        x = torch.sum(x, dim=0, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out
