# encoding:utf-8
import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.Wt = nn.Linear(input_size, input_size)
        self.Wh = nn.Linear(input_size, input_size)

    def forward(self, x):
        t = torch.sigmoid(self.Wt(x))
        return t * torch.relu(self.Wh(x)) + (1-t) * x


class TextCNN(nn.Module):
    def __init__(self, num_label, settings):
        super(TextCNN, self).__init__()

        self.conv_list = nn.ModuleList()
        for kernel_size in settings['text_cnn_filters']:
            conv = nn.Sequential(
                nn.Conv1d(settings['word_emb_dim'], settings['text_cnn_out_dim'],
                          kernel_size, bias=False, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(settings['dropout_p']),
            )
            self.conv_list.append(conv)

        input_dim = len(settings['text_cnn_filters']) * settings['text_cnn_out_dim']
        self.highway = Highway(input_dim)
        self.hw_drop = nn.Dropout(settings['dropout_p'])
        self.fc = nn.Linear(input_dim, num_label)

    def forward(self, word_embs):
        # [bs, seq, word_dim]
        word_embs = word_embs.permute(0, 2, 1)  # [bs, word_dim, seq]
        out = []
        for conv in self.conv_list:
            conv_out = conv(word_embs).squeeze(-1)  # [batch, out_dim]
            out.append(conv_out)
        out = torch.cat(out, dim=-1)  # [bs, num_convs*out_dim]
        out = self.highway(out)  # [bs, num_convs*out_dim]
        out = self.hw_drop(out)
        return self.fc(out)  # [bs, lable_num]
