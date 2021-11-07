# MRC Level 2 Pstage of 청계산셰르파

## 파일 구성

### 저장소 구조

```bash
├── EDA.ipynb
├── README.md
├── Retrieval # Dense(BertEncoder), Sparse(BM25), Hybrid(Dense + Sparse) retrieval 제공
│   ├── caching
│   │   ├── setting.ipynb
│   │   └── setting.py
│   ├── dense_model.py
│   ├── dense_train.py
│   ├── dense_train_utils.py
│   ├── retrieval.py
│   └── retrieval_rerank_biencoder_crossencoder.ipynb # biencoder -> crossencoder를 사용하여 retrieval rerank
├── arguments.py # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음
├── augmentations
│   ├── aeda.py # AEDA Augmentation
│   ├── context_shuffle.ipynb # Context를 shuffle하는 Augmentation
│   └── ner_and_question_generation.ipynb # Pororo의 ner과 question generation을 이용한 Augmentation
├── diff_prediction.py
├── for_submit
│   ├── ensemble.ipynb
│   ├── last_process.py # n_best prediction의 중복된 답의 확률을 합친 결과를 생성하는 파일
│   └── post_process.ipynb
├── inference.py
├── metric.py # 필요한 Metric을 제공하는 파일
├── models
│   ├── cnn_head.py # cnn head를 부착한 backbone
│   ├── frozen_head.py # backbone을 fine tuning한 sota 모델의 parameter를 Freezing하고 head부착
│   └── lstm_roberta.py # roberta에 lstm layer 추가
├── process.py # 데이터를 입력 형식에 맞게 수정해주는 파일
├── setup
│   ├── How_to_use_Retrieval.ipynb # Retrieval 사용법이 적힌 ipython notebook
│   ├── clean_dataset.py # 데이터셋을 전처리하는 코드
│   └── make_train_data_with_concat.ipynb
├── train.py # MRC, Retrieval 모델 학습 및 평가 
└── utils.py # 기타 유틸 함수 제공 
```


## 데이터 소개

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```python
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
    ./new_train_dataset/           # 학습에 사용할 전처리 된 데이터셋. 
    ./preprocess_wiki.json         # 전처리된 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 


## Preprocess
```
python clean_dataset.py # 전처리 된 train/test/wiki 생성
python Retrieval/caching/setting.py # retriever에 필요한 dictionary 생성
```

## Data Augmentation

- `augmentation/aeda.py`
- `augmentation/context_shuffle.ipynb`
- `augmentation/ner_and_question_generation.ipynb`

## Training Dense Retrieval

SparseRetrieval으로 train question, validation question에 대해 top k개의 wiki id들을 찾은 후 인자를 넘겨주어야 합니다.
```
python Retrieval/dense_train.py # dense retriever 생성
```

## 훈련, 추론

### train - default(train with 4 concatenated passages)

make_train_data_with_concat.ipynb를 먼저 실행 
```
# 학습 예시 (train_dataset 사용)
python train.py
```
만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

### inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.


* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

```
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```

### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

## Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 존재하는 파일이 load 됩니다.
2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. ./outputs/ 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.
