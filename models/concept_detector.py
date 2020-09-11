import torch
import torch.nn as nn


class ConceptDetector(nn.Module):
    def __init__(self, idx2concept, settings):
        super(ConceptDetector, self).__init__()
        self.idx2concept = idx2concept

        self.output = nn.Sequential(
            nn.Linear(settings['fc_feat_dim'], settings['concept_mid_him']),
            nn.ReLU(),
            nn.Linear(settings['concept_mid_him'], settings['concept_mid_him']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['concept_mid_him'], len(idx2concept)),
            nn.Sigmoid(),
        )

    def forward(self, features):
        # [bz, fc_feat_dim]
        return self.output(features)  # [bz, num_cpts]

    def sample(self, features, num):
        # [bz, fc_feat_dim]
        self.eval()
        out = self.output(features)  # [bz, num_cpts]
        scores, idx = out.sort(dim=-1, descending=True)
        scores = scores[:, :num]
        idx = idx[:, :num]
        concepts = []
        for batch in idx:
            tmp = []
            for i in batch:
                tmp.append(self.idx2concept[i])
            concepts.append(tmp)
        return out, concepts, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               MultiLabelClsLoss()


class MultiLabelClsLoss(nn.Module):
    def __init__(self):
        super(MultiLabelClsLoss, self).__init__()

    def forward(self, result, target):
        # result/target: [bz, num_cpts]
        target = target.type(result.type())

        output = target * result.log()
        output = - output.mean(dim=-1).mean(dim=-1)  # scalar
        out = (1 - target) * (1 - result).log()
        out = - out.mean(dim=-1).mean(dim=-1)  # scalar

        output += out
        return output
