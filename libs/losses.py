from torch import Tensor
from torch.nn import Module, CosineSimilarity
from torch.nn.functional import relu

esq = 7.3890560989306495


class CustomTripletMarginLoss(Module):
    def __init__(self, margin: float = 1.0, marginIncrement: float = 0.0):
        super(CustomTripletMarginLoss, self).__init__()
        self.marginIncrement = marginIncrement
        self.margin = margin
        self.Cosine = CosineSimilarity()

    def forward(self,
                anchor: Tensor,
                positive: Tensor,
                negative: Tensor,
                epoch: int = -1) -> Tensor:
        if epoch > -1:
            currentMargin = self.margin + self.marginIncrement * epoch
        #if epoch > -1 and epoch % 2 != 0:
        # if epoch is even, then we target dAN
        # dAN > dAP+margin
        # thus we pushing non-siblings aside
        #	return torch.linalg.norm(positive - anchor)
        #else:
        # otherwise (epoch is odd) we want to target dAN
        # dAP->0 is minizing size of cluster
        # by pulling siblings to each other
        #return F.triplet_margin_loss(anchor, positive, negative, margin=currentMargin)
        dAPs = 1 - self.Cosine(anchor, positive)
        dANs = 1 - self.Cosine(anchor, negative)
        losses = relu(dAPs - dANs + currentMargin)
        return losses.mean()  # + mean(dAPs)


class Custom3x3Loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                anchor: Tensor,
                positive: Tensor,
                negative: Tensor,
                epoch: int = -1) -> Tensor:
        if epoch > -1:
            currentMargin = self.margin + self.marginIncrement * epoch
        #if epoch > -1 and epoch % 2 != 0:
        # if epoch is even, then we target dAN
        # dAN > dAP+margin
        # thus we pushing non-siblings aside
        #	return torch.linalg.norm(positive - anchor)
        #else:
        # otherwise (epoch is odd) we want to target dAN
        # dAP->0 is minizing size of cluster
        # by pulling siblings to each other
        #return F.triplet_margin_loss(anchor, positive, negative, margin=currentMargin)
        dAPs = 1 - self.Cosine(anchor, positive)
        dANs = 1 - self.Cosine(anchor, negative)
        losses = relu(dAPs - dANs + currentMargin)
        return losses.mean()  # + mean(dAPs)
