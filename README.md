# Adversarial Scrubber (AdS)

This repository presents the implementation of EMNLP 2021 submission.

## Preparing the data

The dialogue datasets from the paper [Multi-dimensional gender bias classification](https://arxiv.org/pdf/2005.00614.pdf). We use the huggingface library to retrieve the datasets and use the same split.

DIAL and PAN16 datasets were generated using this open-source [project](https://github.com/yanaiela/demog-text-removal).

The Biographies corpus was retrieved from this [project](https://github.com/Microsoft/biosbias).

Run the following command to prepare the data in the format required in our project.
```
python src/prepare_data.py --dataset wizard  --save_path data/wizard/ 
```

## Running AdS

Execute the following commands to run our AdS.

```
python src/train.py \
--dataset wizard \
--MODEL bert-base-uncased \
--max_len 32 \
--embedding_size 768 \
--device cuda:0 \
--model_save_path model/wizard/ \
--results_save_path results/wizard \
--epochs 3 \
--lambda_1 10 \
--lambda_2 1
```

## Evaluating AdS
Evaluate the representations formed by the model by running the following command

```
python src/evaluate.py \
    --train_path results/wizard/train-gen.pkl \
    --test_path results/wizard/test-gen.pkl
```