# 背景

选取中文预训练模型+中文数据集，进行预训练模型加微调方式的实验。

预训练模型：使用huggingface的transformer框架，中文预训练模型，https://huggingface.co/bert-base-chinese

数据集：选择‘机器阅读理解任务’：SQuAD（英文）、DuReader（多文档机器阅读理解）、DRCD（繁体中文单文档机器阅读理解）

框架： https://github.com/huggingface/transformers/tree/master/examples

支持的任务有：问答（机器阅读理解）、分类、文本生成、序列到序列、多选、语言模型


# 1 DuReader
百度机器阅读理解数据集

https://github.com/baidu/DuReader/blob/master/data/download.sh


https://dataset-bj.cdn.bcebos.com/dureader/dureader_raw.zip 1.3G
https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip 2.8G



步骤：
1. 从https://huggingface.co/bert-base-chinese#list-files下载模型文件
2. 模型文件路径：/Users/huihui/data/bert/bert-base-chinese
3. 重命名文件
```
(base) huihui@192 bert-base-chinese % cp bert-base-chinese-config.json config.json
(base) huihui@192 bert-base-chinese % cp bert-base-chinese-vocab.txt vocab.txt
(base) huihui@192 bert-base-chinese % cp bert-base-chinese-pytorch_model.bin pytorch_model.bin
```

## Fine-tuning bert-base-chinese on DuReader


```bash
export SQUAD_DIR=./raw/

python run_squad.py \
  --model_type bert \
  --model_name_or_path /Users/huihui/data/bert/bert-base-chinese \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/trainset/zhidao.train.json \
  --predict_file $SQUAD_DIR/devset/zhidao.dev.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir debug_squad/
```


服务器运行
```
(base) huihui@192 Downloads % scp dureader_raw.zip xuehp@haomeiya002:~/git/transformers/examples/question-answering/
```

- 注意：

DuReader的数据格式和SQUAD不一样。虽然都是问答，但是SQUAD采取的是标注的方式，答案出现在原文中，标注答案的起始位置；而DuReader是直接回答的方式。

而且DuReader是多文档机器阅读理解

# 2 DRCD

## Fine-tuning bert-base-chinese on DRCD

```
/Users/huihui/data/drcd_public/dev.json
```

问题：DRCD是繁体中文，bert-base-chinese是简体中文，所以这个验证失败的概率很大

在GPU机器运行：
`xuehp@haomeiya007:~/git/transformers/examples/question-answering$ `

```

python run_squad.py \
  --model_type bert \
  --model_name_or_path /home/xuehp/data/bert/bert-base-chinese/ \
  --do_train \
  --do_eval \
  --train_file /home/xuehp/data/drcd_public/train.json \
  --predict_file /home/xuehp/data/drcd_public/dev.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir debug_drcd/
```

测试结果：
```
09/09/2020 18:13:58 - INFO - __main__ -   Results: {'exact': 0.031065548306927617, 'f1': 0.05177591384487936, 'total': 3219, 'HasAns_exact': 0.031065548306927617, 'HasAns_f1': 0.05177591384487936, 'HasAns_total': 3219, 'best_exact': 0.031065548306927617, 'best_exact_thresh': 0.0, 'best_f1': 0.05177591384487936, 'best_f1_thresh': 0.0}
```


# 3 cmrc2018_public

CMRC2018, https://github.com/ymcui/cmrc2018

The Second Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC 2018). EMNLP 2019

第二届“讯飞杯”中文机器阅读理解评测（CMRC 2018）

这和SQuAD格式是一样的。

在GPU机器运行：
```
xuehp@haomeiya007:~/git/transformers/examples/question-answering$ `
```

```
python run_squad.py \
  --model_type bert \
  --model_name_or_path /home/xuehp/data/bert/bert-base-chinese/ \
  --do_train \
  --do_eval \
  --train_file /home/xuehp/data/cmrc2018_public/train.json \
  --predict_file /home/xuehp/data/cmrc2018_public/dev.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir debug_squad/
```
测试结果：

```
09/09/2020 17:13:05 - INFO - __main__ -   Results: {'exact': 57.409133271202236, 'f1': 57.85826578930027, 'total': 3219, 'HasAns_exact': 57.409133271202236, 'HasAns_f1': 57.85826578930027, 'HasAns_total': 3219, 'best_exact': 57.409133271202236, 'best_exact_thresh': 0.0, 'best_f1': 57.85826578930027, 'best_f1_thresh': 0.0}
```
以上结果只运行了2个epoch。


加载预训练模型和在数据集微调的具体实现，见`run_squad`
