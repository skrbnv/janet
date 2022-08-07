import torch
from torch.nn import Module, CosineSimilarity
from torch.nn.functional import relu
import numpy as np

esq = 7.3890560989306495


class CustomTripletMarginLoss(Module):
    def __init__(self, margin: float = 1.0, marginIncrement: float = 0.0):
        super().__init__()
        self.marginIncrement = marginIncrement
        self.margin = margin
        self.Cosine = CosineSimilarity()

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor,
                epoch: int = -1) -> torch.Tensor:
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
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor,
                epoch: int = -1) -> torch.Tensor:
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


# Multi-class centroid loss
class CentroidLoss(Module):
    def __init__(self,
                 num_classes,
                 margin_pos=.3,
                 margin_neg=.3,
                 threshold=.8):
        super().__init__()
        self.current_epoch = -1
        self.num_classes = num_classes
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.threshold = threshold
        self.Cosine = CosineSimilarity()
        self.reset()

    def reset(self):
        self.centroids, self.counters = {}, {}

    def append(self, preds, labels):
        indices = torch.where(labels >= self.threshold)
        for rw, cls in zip(indices[0], indices[1]):
            row = rw.item()
            key = cls.item()
            if key not in self.centroids.keys():
                self.centroids[key] = preds[row]
                self.counters[key] = 1
            else:
                c = self.counters[key]
                new_value = c / (c + 1) * self.centroids[key] + 1 / (
                    c + 1) * preds[row]
                self.centroids[key] = new_value
                self.counters[key] = c + 1

    def dist_positive(self, pred, label):
        indices = torch.where(label > 0)[0].tolist()
        centroid_cmb = label[indices[0]] * self.centroids[indices[0]]
        for i in range(1, len(indices)):
            centroid_cmb += label[indices[i]] * self.centroids[indices[i]]
        return 1 - self.Cosine(pred.unsqueeze(0), centroid_cmb.unsqueeze(0))

    def dist_negative(self, pred, label):
        keys = torch.where(label < 1 - self.threshold)[0].tolist()
        output = torch.zeros(1).to(pred.device)
        adds = 0
        for ckey in self.centroids.keys():
            if ckey in keys:
                adds += 1
                dist = 1 - self.Cosine(pred.unsqueeze(0),
                                       self.centroids[ckey].unsqueeze(0))
                loss = torch.max(self.margin_neg - dist,
                                 torch.zeros(1).to(pred.device))
                output += loss
        return output / adds

    def forward(self, preds, labels, reset=True) -> torch.Tensor:

        self.append(preds, labels)

        pos, neg = [], []
        for pred, label in zip(preds, labels):
            pos.append(self.dist_positive(pred, label))
            neg.append(self.dist_negative(pred, label))

        #dAPs = 1 - self.Cosine(anchor, positive)
        #dANs = 1 - self.Cosine(anchor, negative)
        #losses = relu(dAPs - dANs + currentMargin)
        if reset is True:
            self.reset()
        return torch.mean(torch.stack(pos)) + torch.mean(torch.stack(neg))
