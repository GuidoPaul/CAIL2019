#!/usr/bin/python
# coding: utf-8

import json
import pickle
import re

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import QuantileTransformer


def max_min_scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def trans(text):
    text = list(jieba.cut(text))
    text = " ".join([x for x in text if not x.isdigit()])
    return text


def get_sub_col(col):
    sub1_col, sub2_col = [sub for sub in col.split(" \n \n ")]
    return pd.Series([sub1_col, sub2_col])


def get_sub_col_raw(col):
    sub1_col, sub2_col = [sub for sub in col.split("\n\n")]
    return pd.Series([sub1_col, sub2_col])


def get_length_features(df, stopwords, name):
    def words_count(text, stopwords):
        wordlist = [word for word in str(text).split() if word not in stopwords]
        return len(wordlist)

    # Word
    df[f"col_len_{name}"] = df["col"].apply(lambda s: words_count(s, stopwords))
    df[f"sub1_col_len_{name}"] = df["sub1_col"].apply(
        lambda s: words_count(s, stopwords)
    )
    df[f"sub2_col_len_{name}"] = df["sub2_col"].apply(
        lambda s: words_count(s, stopwords)
    )

    df[f"col_len_ratio_{name}"] = (
        df[f"sub1_col_len_{name}"] / df[f"sub2_col_len_{name}"]
    )
    df[f"col_len_ratio2_{name}"] = (
        df[f"sub2_col_len_{name}"] / df[f"col_len_{name}"]
    )

    df[f"col_len_c_{name}"] = df["col"].apply(len)
    df[f"sub1_col_len_c_{name}"] = df["sub1_col"].apply(len)
    df[f"sub2_col_len_c_{name}"] = df["sub2_col"].apply(len)

    df[f"col_len_c_ratio_{name}"] = (
        df[f"sub1_col_len_c_{name}"] / df[f"sub2_col_len_c_{name}"]
    )
    df[f"col_len_c_ratio2_{name}"] = (
        df[f"sub2_col_len_c_{name}"] / df[f"col_len_c_{name}"]
    )

    df[f"sub1_col_len_{name}"] = df[[f"sub1_col_len_{name}"]].apply(
        max_min_scaler
    )
    df[f"sub2_col_len_{name}"] = df[[f"sub2_col_len_{name}"]].apply(
        max_min_scaler
    )
    df[f"sub1_col_len_c_{name}"] = df[[f"sub1_col_len_c_{name}"]].apply(
        max_min_scaler
    )
    df[f"sub2_col_len_c_{name}"] = df[[f"sub2_col_len_c_{name}"]].apply(
        max_min_scaler
    )

    useful_cols = [
        f"sub1_col_len_{name}",
        f"sub2_col_len_{name}",
        f"col_len_ratio_{name}",
        f"col_len_ratio2_{name}",
        #
        f"sub1_col_len_c_{name}",
        f"sub2_col_len_c_{name}",
        f"col_len_c_ratio_{name}",
        f"col_len_c_ratio2_{name}",
    ]

    return df[useful_cols]


def get_plantiff_features(df, name):
    def f_plantiff_is_company(x):
        r = re.search(r"原告(.*?)被告", x)
        s = 0
        if r:
            plantiff = r.group(1)
            if "法定代表人" in plantiff:
                return 1
        return s

    reg = re.compile(r"原告")
    df[f"sub1_col_num_plantiff_{name}"] = df["sub1_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )
    df[f"sub1_col_bool_plantiff_{name}"] = df["sub1_col_raw"].apply(
        lambda s: f_plantiff_is_company(s)
    )

    useful_cols = [
        f"sub1_col_num_plantiff_{name}",
        f"sub1_col_bool_plantiff_{name}",
    ]
    return df[useful_cols]


def get_defendant_features(df, name):
    def f_defandent_noreply(text):
        if any(
            ss in text
            for ss in ["未答辩", "拒不到庭", "未到庭", "未做答辩", "未应诉答辩", "未作出答辩", "未出庭"]
        ):
            return 1
        return 0

    reg = re.compile(r"被告.*?法定代表人.*?。")
    df[f"sub1_col_bool_defendant_{name}"] = df["sub1_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_bool_defendant_noreply_{name}"] = df["sub2_col_raw"].apply(
        lambda s: f_defandent_noreply(s)
    )
    reg = re.compile(r"被告")
    df[f"sub1_col_num_defendant_{name}"] = df["sub1_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )

    useful_cols = [
        f"sub1_col_bool_defendant_{name}",
        f"sub1_col_num_defendant_{name}",
    ]
    return df[useful_cols]


def get_guarantor_features(df, name):
    reg = re.compile(r"担保")
    df[f"sub2_col_bool_guarantor_{name}"] = df["sub2_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_num_guarantor_{name}"] = df["sub2_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )

    useful_cols = [
        f"sub2_col_bool_guarantor_{name}",
        f"sub2_col_num_guarantor_{name}",
    ]
    return df[useful_cols]


def get_guaranty_features(df, name):
    reg = re.compile(r"抵押")
    df[f"sub2_col_bool_guaranty_{name}"] = df["sub2_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_num_guaranty_{name}"] = df["sub2_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )

    useful_cols = [
        f"sub2_col_bool_guarantor_{name}",
        f"sub2_col_num_guarantor_{name}",
    ]
    return df[useful_cols]


def get_interest_features(df, name):
    def do_lixi(text):
        m_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        mm = m_reg.search(text)

        m2_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)毛")
        mm2 = m2_reg.search(text)

        m3_reg = re.compile(r"月(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        mm3 = m3_reg.search(text)

        y_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        ym = y_reg.search(text)

        y2_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)毛")
        ym2 = y2_reg.search(text)

        y3_reg = re.compile(r"年(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        ym3 = y3_reg.search(text)

        if mm:
            return round(float(mm.group(2)) * 12, 2)
        elif mm2:
            return round(float(mm2.group(2)) * 10 * 12, 2)
        elif mm3:
            return round(float(mm3.group(1)) * 12, 2)
        elif ym:
            return float(ym.group(2))
        elif ym2:
            return round(float(ym2.group(2)) * 10, 2)
        elif ym3:
            return float(ym3.group(1))
        else:
            return 0

    def do_lixi_c(text):
        m_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        mm = m_reg.search(text)

        m2_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)毛")
        mm2 = m2_reg.search(text)

        m3_reg = re.compile(r"月(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        mm3 = m3_reg.search(text)

        y_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        ym = y_reg.search(text)

        y2_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)毛")
        ym2 = y2_reg.search(text)

        y3_reg = re.compile(r"年(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        ym3 = y3_reg.search(text)

        count = 0

        if mm:
            count = round(float(mm.group(2)) * 12, 2)
        elif mm2:
            count = round(float(mm2.group(2)) * 10 * 12, 2)
        elif mm3:
            count = round(float(mm3.group(1)) * 12, 2)
        elif ym:
            count = float(ym.group(2))
        elif ym2:
            count = round(float(ym2.group(2)) * 10, 2)
        elif ym3:
            count = float(ym3.group(1))
        else:
            count = 0

        if count == 0:
            return 0
        elif count < 24:
            return 1
        elif count < 36:
            return 2
        else:
            return 3

    reg = re.compile(r"约定利息|约定月利息|年息|月息|利息|利率")
    df[f"sub2_col_bool_interest_{name}"] = df["sub2_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_num_interest_{name}"] = df["sub2_col_raw"].apply(
        lambda s: do_lixi(s)
    )
    df[f"sub2_col_num_interest_c_{name}"] = df["sub2_col_raw"].apply(
        lambda s: do_lixi_c(s)
    )

    useful_cols = [
        f"sub2_col_bool_interest_{name}",
        f"sub2_col_num_interest_{name}",
        f"sub2_col_num_interest_c_{name}",
    ]
    return df[useful_cols]


def get_couple_features(df, name):
    reg = re.compile(r"夫妻")
    df[f"sub2_col_bool_couple_{name}"] = df["sub2_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_num_couple_{name}"] = df["sub2_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )

    useful_cols = [
        f"sub2_col_bool_couple_{name}",
        f"sub2_col_num_couple_{name}",
    ]
    return df[useful_cols]


def get_death_features(df, name):
    reg = re.compile(r"死亡")
    df[f"sub2_col_bool_death_{name}"] = df["sub2_col_raw"].apply(
        lambda s: int(len(reg.findall(s)) > 0)
    )
    df[f"sub2_col_num_death_{name}"] = df["sub2_col_raw"].apply(
        lambda s: len(reg.findall(s))
    )

    useful_cols = [f"sub2_col_bool_death_{name}", f"sub2_col_num_death_{name}"]
    return df[useful_cols]


def do_basic_feature(df, stopwords, name):
    feature_list = []

    feature = get_length_features(df, stopwords, name)
    feature_list.append(feature)

    feature = get_plantiff_features(df, name)
    feature_list.append(feature)

    feature = get_defendant_features(df, name)
    feature_list.append(feature)

    feature = get_guarantor_features(df, name)
    feature_list.append(feature)

    feature = get_guaranty_features(df, name)
    feature_list.append(feature)

    feature = get_interest_features(df, name)
    feature_list.append(feature)

    feature = get_couple_features(df, name)
    feature_list.append(feature)

    index = feature_list[0].index
    for feature_dataset in feature_list[1:]:
        pd.testing.assert_index_equal(index, feature_dataset.index)

    df = pd.concat(feature_list, axis=1)

    return df


def do_tfidf_feature(df, tfidf):
    n_components = 30
    svd = TruncatedSVD(
        n_components=n_components, algorithm="arpack", random_state=2019
    )

    col_tfidf = tfidf.transform(df["col"])

    feature_names = tfidf.get_feature_names()
    ret_df = pd.DataFrame(col_tfidf.toarray(), columns=feature_names)
    return ret_df

    col_svd = svd.fit_transform(col_tfidf)

    best_fearures = [
        feature_names[i] + "i" for i in svd.components_[0].argsort()[::-1]
    ]
    ret_df = pd.DataFrame(col_svd, columns=best_fearures[:n_components])
    return ret_df


def get_length_related_features_col2(df):
    df_copy = df.copy()
    df_copy["col2_len"] = df_copy["col2"].apply(len)
    df_copy["col2_len_relative"] = (
        df_copy["col2_len"] / df_copy["col2_len"].max()
    )

    df_copy["col2_title_len"] = df_copy["col2"].apply(
        lambda s: len(s.split("\n\n")[0])
    )
    df_copy["col2_title_relative"] = (
        df_copy["col2_title_len"] / df_copy["col2_title_len"].max()
    )

    df_copy["col2_content_len"] = (
        df_copy["col2_len"] - df_copy["col2_title_len"]
    )
    df_copy["col2_content_len_relative"] = (
        df_copy["col2_content_len"] / df_copy["col2_content_len"].max()
    )

    df_copy["col2_title_ratio"] = (
        df_copy["col2_title_len"] / df_copy["col2_len"]
    )

    useful_cols = [
        "col2_len_relative",
        "col2_title_relative",
        "col2_content_len_relative",
        "col2_title_ratio",
    ]

    return df_copy[useful_cols]


def get_col2_re_features(df):
    old_cols = set(df.columns)
    # 原告数量, 原告数据差
    reg = re.compile(r"原告")
    df["col2_num_accuser"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    # 原告中男性数量, 比例
    reg = re.compile(r"原告.*?男.*?被告")
    df["col2_num_male_accuser"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    df["col2_num_male_accuser_rate"] = (
        df["col2_num_male_accuser"] / df["col2_num_accuser"]
    )
    # 原告中委托诉讼代理人数量，比例
    reg = re.compile(r"原告.*?委托诉讼代理人.*?被告")
    df["col2_num_company_accuser"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    df["col2_num_company_accuser_rate"] = (
        df["col2_num_company_accuser"] / df["col2_num_accuser"]
    )
    # 被告数量, 原告数据差
    reg = re.compile(r"被告")
    df["col2_num_defendant"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    # 被告中男性数量, 比例
    reg = re.compile(r"被告.*?男.*?。")
    df["col2_num_male_defendant"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    df["col2_num_male_defendant_rate"] = (
        df["col2_num_male_defendant"] / df["col2_num_defendant"]
    )
    df["col2_defendant_minus_num_accuser"] = (
        df["col2_num_defendant"] - df["col2_num_accuser"]
    ).astype(int)
    # 被告中法定代表人数量，比例
    reg = re.compile(r"被告.*?法定代表人.*?。")
    df["col2_num_company_defendant"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[0]))
    )
    df["col2_num_company_accuser_rate"] = (
        df["col2_num_company_defendant"] / df["col2_num_defendant"]
    )
    # 担保
    reg = re.compile(r"担保")
    df["col2_num_danbao"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[1]))
    )

    # 标点符号数量
    reg = re.compile(r"[。：，]")
    df["col2_num_punctuation"] = df["col2"].apply(
        lambda s: len(reg.findall(s.split("\n\n")[1]))
    )
    # 词数量
    df["col2_num_word"] = df["col2"].apply(
        lambda s: len(list(jieba.cut(s.split("\n\n")[1])))
    )
    df["col2_num_word_ratio"] = df["col2_num_word"] / df["col2_num_word"].max()
    df["col2_num_word_divide_length"] = df["col2_num_word"] / df["col2"].apply(
        len
    )

    useful_cols = list(set(df.columns).difference(old_cols))
    return df[useful_cols]


def do_feature_engineering(list_text):
    df = pd.DataFrame(list_text, columns=["col2"])

    feature_list = []
    feature = get_length_related_features_col2(df)
    feature_list.append(feature)
    feature = get_col2_re_features(df)
    feature_list.append(feature)
    index = feature_list[0].index

    for feature_dataset in feature_list[1:]:
        pd.testing.assert_index_equal(index, feature_dataset.index)

    data = pd.concat(feature_list, axis=1)
    qt = QuantileTransformer(random_state=2019)
    for col in data.columns:
        data[col] = qt.fit_transform(data[[col]])
    return data


def do_feature_engineering_bak(list_text, stopwords, name):
    df = pd.DataFrame(list_text, columns=["col_raw"])
    df["col"] = df["col_raw"].apply(trans)

    sub_col_raw_df = df["col_raw"].apply(lambda s: get_sub_col_raw(s))
    sub_col_raw_df.columns = ["sub1_col_raw", "sub2_col_raw"]

    sub_col_df = df["col"].apply(lambda s: get_sub_col(s))
    sub_col_df.columns = ["sub1_col", "sub2_col"]

    df = pd.concat([df, sub_col_raw_df, sub_col_df], axis=1)

    basic_df = do_basic_feature(df, stopwords, name)
    return basic_df


def get_tfidf(train_path, test_path, train_flag=False):
    se = set()
    list_text_a = []
    list_text_b = []
    list_text_c = []
    num_of_train = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            x = json.loads(line)
            se.add(x["A"])
            se.add(x["B"])
            se.add(x["C"])
            if i % 2 == 0:
                list_text_a.append(x["A"])
                list_text_b.append(x["B"])
                list_text_c.append(x["C"])
            else:
                list_text_a.append(x["A"])
                list_text_b.append(x["C"])
                list_text_c.append(x["B"])
            num_of_train += 1
    with open(test_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            x = json.loads(line)
            se.add(x["A"])
            se.add(x["B"])
            se.add(x["C"])
            if i % 2 == 0:
                list_text_a.append(x["A"])
                list_text_b.append(x["B"])
                list_text_c.append(x["C"])
            else:
                list_text_a.append(x["A"])
                list_text_b.append(x["C"])
                list_text_c.append(x["B"])

    df = pd.DataFrame(se, columns=["col_raw"])
    text_a_df = pd.DataFrame(list_text_a, columns=["col_raw"])
    text_b_df = pd.DataFrame(list_text_b, columns=["col_raw"])
    text_c_df = pd.DataFrame(list_text_c, columns=["col_raw"])

    df["col"] = df["col_raw"].apply(trans)
    text_a_df["col"] = text_a_df["col_raw"].apply(trans)
    text_b_df["col"] = text_b_df["col_raw"].apply(trans)
    text_c_df["col"] = text_c_df["col_raw"].apply(trans)

    if train_flag:
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6)
        tfidf.fit(df["col"])
        print("save tfidf ...")
        with open("tfidf.model", "wb") as f:
            pickle.dump(tfidf, f)
    else:
        print("load tfidf ...")
        with open("tfidf.model", "rb") as f:
            tfidf = pickle.load(f)

    text_a_tfidf = tfidf.transform(text_a_df["col"])
    text_b_tfidf = tfidf.transform(text_b_df["col"])
    text_c_tfidf = tfidf.transform(text_c_df["col"])

    n_components = 1024
    svd = TruncatedSVD(
        n_components=n_components, algorithm="arpack", random_state=2019
    )
    if train_flag:
        print(text_a_tfidf.toarray().shape)
        all_text_tfidf = np.concatenate(
            (
                text_a_tfidf.toarray(),
                text_b_tfidf.toarray(),
                text_c_tfidf.toarray(),
            ),
            axis=0,
        )
        print(all_text_tfidf.shape)
        svd.fit(all_text_tfidf)
        text_a_svd = svd.transform(text_a_tfidf)
        text_b_svd = svd.transform(text_b_tfidf)
        text_c_svd = svd.transform(text_c_tfidf)
        print("save svd ...")
        with open("svd.model", "wb") as f:
            pickle.dump(svd, f)
    else:
        print("load svd ...")
        with open("svd.model", "rb") as f:
            svd = pickle.load(f)
        text_a_svd = svd.transform(text_a_tfidf)
        text_b_svd = svd.transform(text_b_tfidf)
        text_c_svd = svd.transform(text_c_tfidf)

    feature_names = tfidf.get_feature_names()
    best_fearures = [
        feature_names[i] for i in svd.components_[0].argsort()[::-1]
    ]
    text_a_svd_df = pd.DataFrame(
        text_a_svd, columns=best_fearures[:n_components]
    )
    text_b_svd_df = pd.DataFrame(
        text_b_svd, columns=best_fearures[:n_components]
    )
    text_c_svd_df = pd.DataFrame(
        text_c_svd, columns=best_fearures[:n_components]
    )

    train_tfidf_a_df = text_a_svd_df[:num_of_train]
    train_tfidf_b_df = text_b_svd_df[:num_of_train]
    train_tfidf_c_df = text_c_svd_df[:num_of_train]
    valid_tfidf_a_df = text_a_svd_df[num_of_train:]
    valid_tfidf_b_df = text_b_svd_df[num_of_train:]
    valid_tfidf_c_df = text_c_svd_df[num_of_train:]

    return (
        train_tfidf_a_df,
        train_tfidf_b_df,
        train_tfidf_c_df,
        valid_tfidf_a_df,
        valid_tfidf_b_df,
        valid_tfidf_c_df,
    )
