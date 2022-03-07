# Adversarial Scrubber (AdS)

This repository presents the implementation of EMNLP 2021 paper: "__[Adversarial Scrubbing of Demographic Information for Text Classification](https://aclanthology.org/2021.emnlp-main.43/)"__

[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/), [Sayan Ghosh](https://sgdgp.github.io/), [Yiyuan Li](https://nativeatom.github.io/), [Junier Oliva](https://cs.unc.edu/person/junier-oliva/), [Shashank Srivastava](https://www.ssriva.com/) and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/).


## Pre-requisities
python 3.8
```
# create a clean conda env
conda create -n adv-scrubber python==3.8 
source activate adv-scrubber

pip install -r requirements.txt
```

## Preparing the data

The dialogue datasets from the paper [Multi-dimensional gender bias classification](https://arxiv.org/pdf/2005.00614.pdf). We use the huggingface library (code present in src/prepare_data.py) to retrieve the datasets and use the same split.

DIAL and PAN16 datasets were generated using this open-source [project](https://github.com/yanaiela/demog-text-removal).

The Biographies corpus was retrieved from this [project](https://github.com/Microsoft/biosbias).

Please cite the respective paper if you're using the any of the above data. 

Demo for the project is provided for Wizard dataset. <br>
Run the following command to prepare the data in the format required in our project.
```
cd src/
python prepare_data.py --dataset wizard  --save_path data/wizard/ 
```

## Running AdS

Execute the following commands to run our AdS.

```
cd src/
python train.py \
--dataset wizard \
--MODEL bert-base-uncased \
--max_len 32 \
--embedding_size 768 \
--device cuda:0 \
--model_save_path model/wizard/ \
--results_save_path results/wizard/ \
--save_path data/wizard \
--epochs 3 \
--lambda_1 10 \
--lambda_2 0
```

## Evaluating AdS
Evaluate the representations formed by the model by running the following command

```
cd src/
python evaluate.py \
    --train_path results/wizard/train-gen.pkl \
    --test_path results/wizard/test-gen.pkl
```

## Citation
```
@inproceedings{basu-roy-chowdhury-etal-2021-adversarial,
    title = "Adversarial Scrubbing of Demographic Information for Text Classification",
    author = "Basu Roy Chowdhury, Somnath  and
      Ghosh, Sayan  and
      Li, Yiyuan  and
      Oliva, Junier  and
      Srivastava, Shashank  and
      Chaturvedi, Snigdha",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.43",
    doi = "10.18653/v1/2021.emnlp-main.43",
    pages = "550--562",
}
```