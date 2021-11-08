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
mrc-level2-nlp-08
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

- Run all the lines in ```elastic_search.ipynb```

- Download and Extract the Pretrained Weight
```
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eMfMzv0gkcTSBQAxtQrFMP_pC5sEeQQq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eMfMzv0gkcTSBQAxtQrFMP_pC5sEeQQq" -O ROOT/mrc-level2-nlp-08/dense_encoder.tar.gz && rm -rf /tmp/cookies.txt
$ tar -xf ROOT/mrc-level2-nlp-08/dense_encoder.tar.gz
```

- Download Wiki Corpus and Klue MRC Dataset
```
$ wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz -O ROOT/data.tar.gz
$ tar -xf ROOT/data.tar.gz
```

- After Preparation
```
data
├──wikipedia_documents.json
├──test_dataset
└──train_dataset
mrc-level2-nlp-08
├──checkpoints
├──dense_encoder
|  └──dense_encoder.pth
├──pickle
|  ├──elastic_training_neg.bin
|  └──wiki_token.bin
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
- wiki_token.bin is automatically produced once the model is run.


### 2. Model Training
```
$ python main.py 
```
- and this will save models with ```.pth``` formats under the ```checkpoints``` folder

### 3. Inference
- There is no ```inference.py```. ```main.py``` creates a score file under scores folder

