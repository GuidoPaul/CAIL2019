#!/usr/bin/python
# coding: utf-8

"""
@Author: shenglin.bsl
@Date: 2019-09-04 20:56:25
@Last Modified by:   shenglin.bsl
@Last Modified time: 2019-09-04 20:56:25
"""

import string

import jieba
from matplotlib import pyplot as plt
from pylab import mpl

from zhon.hanzi import punctuation

# mpl.rcParams["font.sans-serif"] = ["FangSong"]  # 指定默认字体
mpl.rcParams["font.family"] = ["SimHei"]  # 指定默认字体
mpl.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

# regular_punct = list(string.punctuation)
regular_punct = list(set(string.punctuation + punctuation + "U" + "P"))
# regular_punct.remove("[")
# regular_punct.remove("]")

all_key_words = set()


def visualize_tokens_and_weights(ax, tokens_and_weights):
    tokens_text = ""
    pos_x, pos_y = 0, 1
    for index, (token, weight) in enumerate(tokens_and_weights):
        if token in regular_punct or token.isdigit():
            weight = 0
        if weight > 0.01:
            tokens_text += token
            # weight += 0.1
        ax.text(
            pos_x,
            pos_y,
            token,
            # style="italic",
            bbox={"facecolor": "red", "alpha": weight, "pad": 1},
        )
        pos_x = pos_x + 0.04
        pos_y = pos_y
        if pos_x >= 1:
            pos_x = 0
            pos_y -= 0.04

    ax.set_axis_off()

    return tokens_text


def visualize_attention(
    tokens_and_weights_a,
    tokens_and_weights_b,
    tokens_and_weights_c,
    output_path,
):
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0)

    ax1 = fig.add_subplot(131)
    tokens_text_a = visualize_tokens_and_weights(ax1, tokens_and_weights_a)

    ax2 = fig.add_subplot(132)
    tokens_text_b = visualize_tokens_and_weights(ax2, tokens_and_weights_b)

    ax3 = fig.add_subplot(133)
    tokens_text_c = visualize_tokens_and_weights(ax3, tokens_and_weights_c)

    cut_tokens_text_a = jieba.cut(tokens_text_a)
    cut_tokens_text_b = jieba.cut(tokens_text_b)
    cut_tokens_text_c = jieba.cut(tokens_text_c)

    setaNb = set(cut_tokens_text_a).intersection(set(cut_tokens_text_b))
    intersection_aNb = list(setaNb)
    intersection_aUb_c = list(setaNb - (set(cut_tokens_text_c)))

    for word in intersection_aNb:
        if len(word) >= 2:
            all_key_words.add(word)

    intersection_key_abc = []
    for word in intersection_aUb_c:
        if len(word) >= 2:
            intersection_key_abc.append(word)

    ax1.text(
        0,
        0.2,
        intersection_key_abc,
        bbox={"facecolor": "red", "alpha": 0, "pad": 1},
    )

    plt.savefig(output_path, bbox_inches="tight")

    return all_key_words
