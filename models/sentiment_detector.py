import torch
import torch.nn as nn


class SentimentDetector(nn.Module):
    def __init__(self, sentiment_categories, settings):
        super(SentimentDetector, self).__init__()
        self.sentiment_categories = sentiment_categories

        self.convs = nn.Sequential()
        in_channels = settings['fc_feat_dim']
        for i in range(settings['sentiment_convs_num']):
            self.convs.add_module(
                'conv_%d' % i, nn.Conv2d(in_channels, in_channels // 2, 3, padding=1))
            in_channels //= 2
        self.convs.add_module('dropout', nn.Dropout(settings['dropout_p']))
        self.convs.add_module('relu', nn.ReLU())

        # TODO: Can be modified for multiple kernels per sentiment
        num_sentis = len(sentiment_categories)
        self.senti_conv = nn.Conv2d(in_channels, num_sentis, 1)
        # self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # self.output = nn.Sequential(
        #     *[nn.Linear(num_sentis, num_sentis) for _ in settings['sentiment_fcs_num']]
        # )
        self.output = nn.Linear(num_sentis, num_sentis)
        # self.output1 = nn.Linear(14*14, num_sentis)

    def forward(self, features):
        # [bz, 14, 14, fc_feat_dim]
        features = features.permute(0, 3, 1, 2)  # [bz, fc_feat_dim, 14, 14]
        features = self.convs(features)  # [bz, channels, 14, 14]
        senti_features = self.senti_conv(features)  # [bz, num_sentis, 14, 14]
        features = self.global_pool(senti_features)  # [bz, num_sentis, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [bz, num_sentis]
        output = self.output(features)  # [bz, num_sentis]

        out = output.softmax(dim=-1)  # [bz, num_sentis]
        shape = senti_features.shape
        senti_features = out.unsqueeze(1).bmm(
            senti_features.view(shape[0], shape[1], -1))  # [bz, 1, 14*14]
        senti_features = senti_features.view(shape[0], shape[2], shape[3])  # [bz, 14, 14]

        # output1 = self.output1(senti_featuresview(shape[0], -1))  # [bz, 1, num_sentis]
        # output1 = output1.squeeze(1)  # [bz, num_sentis]
        # output += output1  # [bz, num_sentis], TODO: or output two results respectively

        return output, senti_features

    def sample(self, features):
        # [bz, 14, 14, fc_feat_dim]
        self.eval()
        output, senti_features = self.forward(features)
        output = output.softmax(dim=-1)
        scores, idx = output.max(dim=-1)  # bz
        sentiments = []
        for i in idx:
            sentiments.append(self.sentiment_categories[i])

        return idx, senti_features, sentiments, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               nn.CrossEntropyLoss()
