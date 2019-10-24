#!/usr/bin/python
# coding: utf-8

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss


class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout1D, self).__init__()
        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, maxlen, 1, embed_size)
        x = x.permute(0, 3, 2, 1)  # (N, embed_size, 1, maxlen)
        x = self.dropout2d(x)  # (N, embed_size, 1, maxlen)
        x = x.permute(0, 3, 2, 1)  # (N, maxlen, 1, embed_size)
        x = x.squeeze(2)  # (N, maxlen, embed_size)

        return x


LSTM_UNITS = 128
CHANNEL_UNITS = 64


class BertCNNForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCNNForTripletNet, self).__init__(config)

        filters = [3, 4, 5]

        self.bert = BertModel(config)
        self.embedding_dropout = SpatialDropout1D(config.hidden_dropout_prob)

        self.conv_layers = nn.ModuleList()
        for filter_size in filters:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size,
                    CHANNEL_UNITS,
                    kernel_size=filter_size,
                    padding=1,
                ),
                # nn.BatchNorm1d(CHANNEL_UNITS),
                # nn.ReLU(inplace=True),
            )
            self.conv_layers.append(conv_block)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )
        # h_embedding_a = self.embedding_dropout(bert_output_a)
        # h_embedding_b = self.embedding_dropout(bert_output_b)
        # h_embedding_c = self.embedding_dropout(bert_output_c)
        h_embedding_a = bert_output_a
        h_embedding_b = bert_output_b
        h_embedding_c = bert_output_c

        h_embedding_a = h_embedding_a.permute(0, 2, 1)
        feature_maps_a = []
        for layer in self.conv_layers:
            h_x_a = layer(h_embedding_a)
            feature_maps_a.append(
                F.max_pool1d(h_x_a, kernel_size=h_x_a.size(2)).squeeze()
            )
            feature_maps_a.append(
                F.avg_pool1d(h_x_a, kernel_size=h_x_a.size(2)).squeeze()
            )
        conv_features_a = torch.cat(feature_maps_a, 1)

        h_embedding_b = h_embedding_b.permute(0, 2, 1)
        feature_maps_b = []
        for layer in self.conv_layers:
            h_x_b = layer(h_embedding_b)
            feature_maps_b.append(
                F.max_pool1d(h_x_b, kernel_size=h_x_b.size(2)).squeeze()
            )
            feature_maps_b.append(
                F.avg_pool1d(h_x_b, kernel_size=h_x_b.size(2)).squeeze()
            )
        conv_features_b = torch.cat(feature_maps_b, 1)

        h_embedding_c = h_embedding_c.permute(0, 2, 1)
        feature_maps_c = []
        for layer in self.conv_layers:
            h_x_c = layer(h_embedding_c)
            feature_maps_c.append(
                F.max_pool1d(h_x_c, kernel_size=h_x_c.size(2)).squeeze()
            )
            feature_maps_c.append(
                F.avg_pool1d(h_x_c, kernel_size=h_x_c.size(2)).squeeze()
            )
        conv_features_c = torch.cat(feature_maps_c, 1)

        h_conc_a = torch.cat((conv_features_a, pooled_output_a), 1)
        h_conc_b = torch.cat((conv_features_b, pooled_output_b), 1)
        h_conc_c = torch.cat((conv_features_c, pooled_output_c), 1)

        return h_conc_a, h_conc_b, h_conc_c


class BertLSTMForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLSTMForTripletNet, self).__init__(config)

        self.bert = BertModel(config)

        self.lstm = nn.LSTM(
            config.hidden_size, 30, bidirectional=True, batch_first=True
        )
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )

        h_lstm_a, (hidden_state_a, cell_state_a) = self.lstm(bert_output_a)
        hh_lstm_a = torch.cat((hidden_state_a[0], hidden_state_a[1]), dim=1)
        avg_pool_a = torch.mean(h_lstm_a, 1)
        max_pool_a, _ = torch.max(h_lstm_a, 1)

        h_lstm_b, (hidden_state_b, cell_state_b) = self.lstm(bert_output_b)
        hh_lstm_b = torch.cat((hidden_state_b[0], hidden_state_b[1]), dim=1)
        avg_pool_b = torch.mean(h_lstm_b, 1)
        max_pool_b, _ = torch.max(h_lstm_b, 1)

        h_lstm_c, (hidden_state_c, cell_state_c) = self.lstm(bert_output_c)
        hh_lstm_c = torch.cat((hidden_state_c[0], hidden_state_c[1]), dim=1)
        avg_pool_c = torch.mean(h_lstm_c, 1)
        max_pool_c, _ = torch.max(h_lstm_c, 1)

        h_conc_a = torch.cat(
            (avg_pool_a, hh_lstm_a, max_pool_a, pooled_output_a), 1
        )
        h_conc_b = torch.cat(
            (avg_pool_b, hh_lstm_b, max_pool_b, pooled_output_b), 1
        )
        h_conc_c = torch.cat(
            (avg_pool_c, hh_lstm_c, max_pool_c, pooled_output_c), 1
        )

        return h_conc_a, h_conc_b, h_conc_c


class BertLSTMGRUForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLSTMGRUForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        # self.embedding_dropout = SpatialDropout1D(config.hidden_dropout_prob)

        self.lstm = nn.LSTM(
            config.hidden_size, LSTM_UNITS, bidirectional=True, batch_first=True
        )
        self.gru = nn.GRU(
            LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True
        )
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )
        h_lstm_a, _ = self.lstm(bert_output_a)
        h_gru_a, hh_gru_a = self.gru(h_lstm_a)
        hh_gru_a = hh_gru_a.view(-1, 2 * LSTM_UNITS)
        avg_pool_a = torch.mean(h_gru_a, 1)
        max_pool_a, _ = torch.max(h_gru_a, 1)

        h_lstm_b, _ = self.lstm(bert_output_b)
        h_gru_b, hh_gru_b = self.gru(h_lstm_b)
        hh_gru_b = hh_gru_b.view(-1, 2 * LSTM_UNITS)
        avg_pool_b = torch.mean(h_gru_b, 1)
        max_pool_b, _ = torch.max(h_gru_b, 1)

        h_lstm_c, _ = self.lstm(bert_output_c)
        h_gru_c, hh_gru_c = self.gru(h_lstm_c)
        hh_gru_c = hh_gru_c.view(-1, 2 * LSTM_UNITS)
        avg_pool_c = torch.mean(h_gru_c, 1)
        max_pool_c, _ = torch.max(h_gru_c, 1)

        h_conc_a = torch.cat(
            (avg_pool_a, hh_gru_a, max_pool_a, pooled_output_a), 1
        )
        h_conc_b = torch.cat(
            (avg_pool_b, hh_gru_b, max_pool_b, pooled_output_b), 1
        )
        h_conc_c = torch.cat(
            (avg_pool_c, hh_gru_c, max_pool_c, pooled_output_c), 1
        )

        return h_conc_a, h_conc_b, h_conc_c


class BertFtsForTripletNet(BertPreTrainedModel):
    def __init__(self, config, num_fts=19):
        super(BertFtsForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.fc = nn.Linear(num_fts, 19)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        x_a,
        x_b,
        x_c,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        _, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        _, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        _, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )

        out_a = self.fc(x_a)
        out_b = self.fc(x_b)
        out_c = self.fc(x_c)

        h_conc_a = torch.cat((pooled_output_a, out_a), 1)
        h_conc_b = torch.cat((pooled_output_b, out_b), 1)
        h_conc_c = torch.cat((pooled_output_c, out_c), 1)

        return h_conc_a, h_conc_b, h_conc_c


class BertTwoForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertTwoForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a1,
        input_ids_b1,
        input_ids_c1,
        token_type_ids_a1=None,
        token_type_ids_b1=None,
        token_type_ids_c1=None,
        attention_mask_a1=None,
        attention_mask_b1=None,
        attention_mask_c1=None,
        input_ids_a2=None,
        input_ids_b2=None,
        input_ids_c2=None,
        token_type_ids_a2=None,
        token_type_ids_b2=None,
        token_type_ids_c2=None,
        attention_mask_a2=None,
        attention_mask_b2=None,
        attention_mask_c2=None,
    ):
        _, pooled_output_a1 = self.bert(
            input_ids=input_ids_a1,
            token_type_ids=token_type_ids_a1,
            attention_mask=attention_mask_a1,
            output_all_encoded_layers=False,
        )
        _, pooled_output_b1 = self.bert(
            input_ids=input_ids_b1,
            token_type_ids=token_type_ids_b1,
            attention_mask=attention_mask_b1,
            output_all_encoded_layers=False,
        )
        _, pooled_output_c1 = self.bert(
            input_ids=input_ids_c1,
            token_type_ids=token_type_ids_c1,
            attention_mask=attention_mask_c1,
            output_all_encoded_layers=False,
        )
        _, pooled_output_a2 = self.bert2(
            input_ids=input_ids_a2,
            token_type_ids=token_type_ids_a2,
            attention_mask=attention_mask_a2,
            output_all_encoded_layers=False,
        )
        _, pooled_output_b2 = self.bert2(
            input_ids=input_ids_b2,
            token_type_ids=token_type_ids_b2,
            attention_mask=attention_mask_b2,
            output_all_encoded_layers=False,
        )
        _, pooled_output_c2 = self.bert2(
            input_ids=input_ids_c2,
            token_type_ids=token_type_ids_c2,
            attention_mask=attention_mask_c2,
            output_all_encoded_layers=False,
        )
        h_conc_a = torch.cat((pooled_output_a1, pooled_output_a2), 1)
        h_conc_b = torch.cat((pooled_output_b1, pooled_output_b2), 1)
        h_conc_c = torch.cat((pooled_output_c1, pooled_output_c2), 1)

        return h_conc_a, h_conc_b, h_conc_c


class BertEmbedding2ForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEmbedding2ForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=True,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=True,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=True,
        )
        last_cat_a = torch.cat(
            (
                bert_output_a[-1][:, 0],
                bert_output_a[-2][:, 0],
                bert_output_a[-3][:, 0],
            ),
            1,
        )
        last_cat_b = torch.cat(
            (
                bert_output_b[-1][:, 0],
                bert_output_b[-2][:, 0],
                bert_output_b[-3][:, 0],
            ),
            1,
        )
        last_cat_c = torch.cat(
            (
                bert_output_c[-1][:, 0],
                bert_output_c[-2][:, 0],
                bert_output_c[-3][:, 0],
            ),
            1,
        )

        return last_cat_a, last_cat_b, last_cat_c


class BertEmbeddingForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEmbeddingForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=True,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=True,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=True,
        )
        last_cat_a = torch.cat(
            (pooled_output_a, bert_output_a[-1][:, 0], bert_output_a[-2][:, 0]),
            1,
        )
        last_cat_b = torch.cat(
            (pooled_output_b, bert_output_b[-1][:, 0], bert_output_b[-2][:, 0]),
            1,
        )
        last_cat_c = torch.cat(
            (pooled_output_c, bert_output_c[-1][:, 0], bert_output_c[-2][:, 0]),
            1,
        )

        return last_cat_a, last_cat_b, last_cat_c


class BertPoolForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPoolForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )
        avg_pool_a = torch.mean(bert_output_a, 1)
        avg_pool_b = torch.mean(bert_output_b, 1)
        avg_pool_c = torch.mean(bert_output_c, 1)

        h_conc_a = torch.cat((avg_pool_a, pooled_output_a), 1)
        h_conc_b = torch.cat((avg_pool_b, pooled_output_b), 1)
        h_conc_c = torch.cat((avg_pool_c, pooled_output_c), 1)

        return h_conc_a, h_conc_b, h_conc_c


class BertNormForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNormForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.norm = nn.BatchNorm1d(config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        bert_output_b, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        bert_output_c, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )

        normed_output_a = self.norm(pooled_output_a)
        normed_output_b = self.norm(pooled_output_b)
        normed_output_c = self.norm(pooled_output_c)

        return normed_output_a, normed_output_b, normed_output_c


class BertForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTripletNet, self).__init__(config)

        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        input_ids_c,
        token_type_ids_a=None,
        token_type_ids_b=None,
        token_type_ids_c=None,
        attention_mask_a=None,
        attention_mask_b=None,
        attention_mask_c=None,
    ):
        _, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )
        _, pooled_output_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
            output_all_encoded_layers=False,
        )
        _, pooled_output_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
            output_all_encoded_layers=False,
        )

        return pooled_output_a, pooled_output_b, pooled_output_c
