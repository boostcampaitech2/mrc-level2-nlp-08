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

## Dataset and Weights for Pretrained Models


<br/>

## Function Description
`main.py`: main module that combines and runs all other sub-modules

`train.py`: trains the model by iterating through specific number of epochs

`model.py`: EfficientNet model from [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)

`utils.py`: required by EfficientNet model

`inference.py`: tests the model using the test dataset and outputs the inferred csv file

`loss.py`: calculates loss using cross entropy and f1-score

`label_smoothing_loss.py`: calculates loss using cross entropy with label smoothing and f1-score

`dataset.py`: generates the dataset to feed the model to train

`data_reset.py`: generates the image dataset divided into 18 classes (train and validation)

`early_stopping.py`: Early Stopping function from [Bjarten](https://github.com/Bjarten/early-stopping-pytorch) (patience decides how many epochs to tolerate after val loss exceeds min. val loss)

`transformation.py`: a group of transformation functions that can be claimed by args parser

`dashboard.ipynb`: can observe the images with labels from the inferred csv files
<br/><br/>

## USAGE
### 1. Data Generation

- Before Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        └──train.csv
```

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


