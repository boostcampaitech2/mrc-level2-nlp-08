# Model for Passage Retrieval

## Main Subject
A dense retrieval model for Open Domain Question Answering (ODQA) task. Two different models encode passages and queries respectively, and we apply dot product to the encoded vectors to calculate the score. This score is then combined with the sparse retrieval score from elastic search to find the most adequate passage for the task.

## Installation
**1. Set up the python environment:**
- Recommended python version 3.8.5
```
$ conda create -n "name of virtual environment" python=3.8.5 pip
$ conda activate "name of virtual environment"
```
**2. Install other required packages**
```
$ pip install -r $ROOT/mrc-level2-nlp-08/requirements.txt
```
<br/>

## Function Description
`main.py`: main module that combines and runs all other sub-modules

`train.py`: trains the model by iterating through specific number of epochs. Since this model does not require minimal time for inference, we create inferences at every epoch with higher validation accuracy.

`model.py`: defines Multilingual BERT class, which we will use with pretrained weights

`arguments.py`: defines the arguements we can use for training. Uses argparser.

`elasticsearch.ipynb`: for sampling negative samples for training. Uses elastic search results to pick pharagraphs with high scores without ground truth results.

`utils.py`: aid function for train.py and main.py
<br/><br/>

## Preparing Dataset and Weights for Pretrained Models


## USAGE
### 1. Preparing Dataset and Weights for Pretrained Models

- Before Preparation
```
data
└──
input
└──mrc-level2-nlp-08
    ├──dense_encoder
    ├──pickle
    ├──scores
    ├──.gitignore
    ├──arguments.py
    ├──elastic_search.ipynb
    ├──main.py
    ├──model.py
    ├──readme.md
    ├──train.py
    └──utils.py
```

- Run all the lines in 
```elastic_search.ipynb```


- Run python file to generate mask classification datasets
```
$ python data_reset.py
```

- After Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```

### 2. Model Training

- Early stopping applied by (default) 

```
$ python main.py --model 7 --tf yogurt --lr 2e-3 --batch_size 16 --num_workers 4 --patience 10 --cut_mix --epochs 100
```

**Image Transformation**<br>
- argument parser `--tf` can receive types of augmentation
- Transformation functions applied to training datasets and test datasets are different: images for inference should be modified as little as possible

- Consult [transformation.py](https://github.com/boostcampaitech2/image-classification-level1-30/blob/main/transformation.py) for detailed explanation on the types of transformation

### 3. Inference
```
$ python inference.py --tf yogurt
```
- Running the line above will generate submission.csv as below

```
input
└──data
    ├──eval
    |  ├──images/
    |  ├──submission.csv
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```


