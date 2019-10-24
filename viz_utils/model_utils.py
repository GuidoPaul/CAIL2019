#!/usr/bin/python
# coding: utf-8

"""
@Author: shenglin.bsl
@Date: 2019-09-04 17:56:27
@Last Modified by:   shenglin.bsl
@Last Modified time: 2019-09-04 17:56:27
"""

import re

import torch


def get_tokenized_text(tokenizer, input_ids):
    input_ids = input_ids
    sentence = tokenizer.decode(token_ids=input_ids)
    if isinstance(sentence, list):
        sentence = sentence[0]

    sentence = sentence.replace("[CLS]", "C")
    sentence = sentence.replace("[SEP]", "E")
    sentence = sentence.replace("[UNK]", "U")
    sentence = sentence.replace("[PAD]", "P")
    sentence = sentence.lstrip().rstrip()
    # for punct in regular_punct:
    #     sentence = sentence.replace(punct, f" {punct} ")
    sentence = re.sub(" +", " ", sentence)
    sentence = "".join(sentence.split())

    return sentence[1:-1]


def get_attention_nth_layer_mth_head_kth_token(
    attention_outputs, n, m, k, average_heads=False, logger=None
):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the nth multi-head attention layer assigned to kth token
    ii)  Take the mth attention head
    """
    if average_heads is True and m is not None:
        logger.warning(
            "Argument passed for param @m will be ignored because of head averaging."
        )

    # Get the attention weights outputted by the nth layer
    attention_outputs_concatenated = torch.cat(
        attention_outputs, dim=0
    )  # (K, N, P, P) (12, 12, P, P)
    attention_outputs = attention_outputs_concatenated.data[
        n, :, :, :
    ]  # (N, P, P) (12, P, P)

    # Get the attention weights assigned to kth token
    attention_outputs = attention_outputs[:, k, :]  # (N, P) (12, P)

    # Compute the average attention weights across all attention heads
    if average_heads:
        attention_outputs = torch.sum(attention_outputs, dim=0)  # (P)
        num_attention_heads = attention_outputs_concatenated.shape[1]
        attention_outputs /= num_attention_heads
    # Get the attention weights of mth head
    else:
        attention_outputs = attention_outputs[m, :]  # (P)

    return attention_outputs


def get_normalized_attention(
    model,
    input_ids,
    token_type_ids,
    attention_mask,
    method="last_layer_heads_average",
    normalization_method="normal",
    device=None,
    logger=None,
):
    """
    Function to get the normalized version of the attention output
    """
    assert method in (
        "first_layer_heads_average",
        "last_layer_heads_average",
        "nth_layer_heads_average",
        "nth_layer_mth_head",
        "custom",
    )
    assert normalization_method in ("normal", "min-max")

    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(data=input_ids, device=device)
        token_type_ids = torch.tensor(data=token_type_ids, device=device)
        attention_mask = torch.tensor(data=attention_mask, device=device)
        input_ids = input_ids.unsqueeze(dim=1).view(1, -1)
        token_type_ids = token_type_ids.unsqueeze(dim=1).view(1, -1)
        attention_mask = attention_mask.unsqueeze(dim=1).view(1, -1)

        output = model(input_ids, token_type_ids, attention_mask)

        attention_outputs = output[-1]  # tuple, 12, [0]: (1, 12, P, P)

    attention_weights = None
    if method == "first_layer_heads_average":
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=0,
            m=None,
            k=0,
            average_heads=True,
            logger=logger,
        )
    elif method == "last_layer_heads_average":
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=-1,
            m=None,
            k=0,
            average_heads=True,
            logger=logger,
        )
    elif method == "nth_layer_heads_average":
        n = -1
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=None,
            k=0,
            average_heads=True,
            logger=logger,
        )
    elif method == "nth_layer_mth_head":
        n = -1
        m = -1
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=m,
            k=0,
            average_heads=False,
            logger=logger,
        )
    elif method == "custom":
        n = -1
        m = -1
        k = 0
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=m,
            k=k,
            average_heads=False,
            logger=logger,
        )

    # Remove the beginning [CLS] & ending [SEP] tokens for better intuition
    attention_weights = attention_weights[1:-1]

    # Apply normalization methods to attention weights
    if normalization_method == "min-max":  # Min-Max Normalization
        max_weight, min_weight = (
            attention_weights.max(),
            attention_weights.min(),
        )
        attention_weights = (attention_weights - min_weight) / (
            max_weight - min_weight
        )
    elif normalization_method == "normal":  # Z-Score Normalization
        mu, std = attention_weights.mean(), attention_weights.std()
        attention_weights = (attention_weights - mu) / std

    # Convert tensor to NumPy array
    attention_weights = attention_weights.data

    return attention_weights


# def get_bert_attention(model, tokenizer, sentence, device):
#     """Function for getting the multi-head self-attention output from pretrained BERT"""
#     # Tokenize & encode raw sentence
#     x = tokenize_and_encode(
#         text=sentence,  # (P)
#         tokenizer=tokenizer,
#         max_tokenization_length=self.config.max_position_embeddings,  # 512
#         truncation_method="head-only",
#     )
#     # Convert the tokenized list to a Tensor
#     x = torch.tensor(data=x, device=device)
#     # Reshape input for BERT output
#     x = x.unsqueeze(dim=1).view(1, -1)  # (B=1, P)
#     # Get features
#     token_type_ids, attention_mask = get_features(
#         input_ids=x, tokenizer=tokenizer, device=device
#     )
#     # Pass tokenized sequence through pretrained BERT model
#     bert_outputs = model(
#         input_ids=x,  # (...) SEE forward()
#         token_type_ids=token_type_ids,
#         attention_mask=attention_mask,
#         position_ids=None,
#         head_mask=None,
#     )
#     attention_outputs = bert_outputs[3]  # ([K] x (1, N, P, P))
#     return attention_outputs
