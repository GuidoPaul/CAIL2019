# 中国法研杯相似案例匹配Top1团队解决方案

## 背景介绍

## 赛题介绍

## 赛题成绩

## 项目构建

### 运行环境

#### 软件依赖
* Python 3.6+
* PyTorch 1.1.0+
* requirements.txt
* Windows 和 Linux 均可

#### 切分数据集
```
$ cd datasets
$ python ./split_folds.py
$ cd ..
```

### 模型训练
```
$ python train_bert.py
```

### 模型预测
```
$ python main.py
```