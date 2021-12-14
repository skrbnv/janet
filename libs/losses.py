from torch import Tensor, is_tensor
from torch.nn import Module, CosineSimilarity
from torch.nn.functional import relu
import numpy as np

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


class Losses():
    def __init__(self) -> None:
        self.losses = {}

    def append(self, losses, epoch=None):
        assert epoch is not None, 'No epoch value provided'
        if epoch in self.losses.keys():
            self.losses[epoch].extend(losses)
        else:
            self.losses[epoch] = losses

    def list(self, epoch=None):
        assert epoch is not None, 'No epoch value provided'
        if epoch in self.losses.keys():
            return self.losses[epoch]
        else:
            return None

    def mean(self, epoch=None):
        return np.mean(np.array(self.list(epoch)))

    def min(self, epoch=None):
        return np.min(np.array(self.list(epoch)))

    def max(self, epoch=None):
        return np.max(np.array(self.list(epoch)))

    def mean_per_epoch(self):
        losses = [self.mean(key) for key in self.losses.keys()]
        return losses
