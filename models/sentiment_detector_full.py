import torch
import torch.nn as nn


class SentimentDetector(nn.Module):
    def __init__(self, sentiment_categories, settings):
        super(SentimentDetector, self).__init__()
        self.sentiment_categories = sentiment_categories
        self.neu_idx = sentiment_categories.index('neutral')

        self.convs = nn.Sequential()
        in_channels = settings['fc_feat_dim']
        for i in range(settings['sentiment_convs_num']):
            self.convs.add_module(
                'conv_%d' % i, nn.Conv2d(in_channels, in_channels // 2, 3, padding=1))
            in_channels //= 2
        self.convs.add_module('dropout', nn.Dropout(settings['dropout_p']))
        self.convs.add_module('relu', nn.ReLU())

        num_kernels_per_sentiment = settings['num_kernels_per_sentiment']
        # TODO: Can be modified for multiple kernels per sentiment
        num_sentis = len(sentiment_categories)
        self.senti_conv = nn.Conv2d(in_channels, num_sentis * num_kernels_per_sentiment, 1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.senti_pool = nn.AdaptiveAvgPool1d(num_sentis)

        self.senti_feat_pool = nn.AdaptiveAvgPool1d(num_sentis)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.cls = nn.Linear(2 * in_channels, num_sentis)

    def forward(self, features):
        # [bz, 14, 14, fc_feat_dim]
        features = features.permute(0, 3, 1, 2)  # [bz, fc_feat_dim, 14, 14]
        features = self.convs(features)  # [bz, n, 14, 14]
        senti_features = self.senti_conv(features)  # [bz, k*C, 14, 14]
        pooled_vector = self.global_max_pool(senti_features)  # [bz, k*C, 1, 1]
        pooled_vector = pooled_vector.squeeze(-1).permute(0, 2, 1)  # [bz, 1, k*C]
        pooled_vector = self.senti_pool(pooled_vector)  # [bz, 1, C]
        det_out = pooled_vector.squeeze(1)  # [bz, C]

        weights = pooled_vector.softmax(dim=-1)  # [bz, 1, C]
        shape = senti_features.shape
        senti_features = senti_features.reshape(shape[0], shape[1], -1).permute(0, 2, 1)  # [bz, 14*14, k*C]
        senti_features = self.senti_feat_pool(senti_features)  # [bz, 14*14, C]
        senti_features = weights.bmm(senti_features.permute(0, 2, 1))  # [bz, 1, 14*14]
        senti_features = senti_features.reshape(shape[0], 1, shape[2], shape[3])  # [bz, 1, 14, 14]

        semantic_features = torch.cat([features, features * senti_features.expand_as(features)], dim=1)  # [bz, 2n, 14, 14]
        semantic_features = self.global_avg_pool(semantic_features)  # [bz, 2n, 1, 1]
        semantic_features = semantic_features.squeeze(-1).squeeze(-1)  # [bz, 2n]
        cls_out = self.cls(semantic_features)  # [bz, C]

        return (det_out, cls_out), senti_features.squeeze(1)

    def sample(self, features, senti_threshold=0):
        # [bz, 14, 14, n]
        self.eval()
        (_, output), senti_features = self.forward(features)
        output = output.softmax(dim=-1)
        scores, senti_labels = output.max(dim=-1)  # bz
        replace_idx = (scores < senti_threshold).nonzero().view(-1)
        senti_labels.index_copy_(0, replace_idx, senti_labels.new_zeros(len(replace_idx)).fill_(self.neu_idx))

        sentiments = []
        for i in senti_labels:
            sentiments.append(self.sentiment_categories[i])

        return senti_labels, senti_features, sentiments, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               nn.CrossEntropyLoss()
