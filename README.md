# Generation Based MRC Using KoBART

## 1. 실행 방법
1. 올바른 경로에 dataset을 받아주고 clean_dataset.py를 실행해 후처리된 dataset을 얻습니다.
    ~~~
    !python clean_dataset.py
    ~~~

2. arguments.py에 있는 학습에 필요한 인자들을 목적에 맞게 수정해줍니다.
3. train.py를 통해 KoBART를 학습시킵니다. Epoch마다 generation.json, prediction.json이 산출되며, 이를 통해 모델이 Evaluation Set에 대해 예측한 답안을 확인할 수 있습니다.<br/>Validation Set에 대해 EM(Exact Match)이 높은 모델이 Best Model 선택됩니다. 이는 arguments.py의 metric_for_best_model을 수정하므로써 변경 가능합니다.
    ~~~
    !python train.py
4. inference.py를 실행해 Test Set에 대한 추론을 진행합니다.<br/> 위와 같이 generation.json을 통해 모델이 Context window만큼의 Passage를 기반으로 생성해낸 답안과, 생성해낸 답안에 대한 Decoder의 생성과정에 있어서 산출된 로짓 기반 Score를 확인할 수 있습니다.<br/>
    ~~~
    !python inference.py
    ~~~
    ~~~
    