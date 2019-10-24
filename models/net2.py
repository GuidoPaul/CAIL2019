#!/usr/bin/python
# coding: utf-8

from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTripletNet, self).__init__(config)
        self.bert = BertModel(config)

        # self.apply(self.init_weights)
        self.init_weights()

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
        outputs_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
        )
        outputs_b = self.bert(
            input_ids=input_ids_b,
            token_type_ids=token_type_ids_b,
            attention_mask=attention_mask_b,
        )
        outputs_c = self.bert(
            input_ids=input_ids_c,
            token_type_ids=token_type_ids_c,
            attention_mask=attention_mask_c,
        )
        _, pooled_output_a, _ = (outputs_a[0], outputs_a[1], outputs_a[-1])
        _, pooled_output_b, _ = (outputs_b[0], outputs_b[1], outputs_b[-1])
        _, pooled_output_c, _ = (outputs_c[0], outputs_c[1], outputs_c[-1])

        return pooled_output_a, pooled_output_b, pooled_output_c
