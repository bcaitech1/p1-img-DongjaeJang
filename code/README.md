# my_pipeline_package 설명

## Components

### dataset.py
- 학습 또는 테스트에 필요한 데이터셋들을 미리 지정해놓은 component

    1. TrainDataset
        - 강의에서 배웠듯 모든 train 폴더의 이미지를 구분없이 모두 라벨링해서 한 개의 데이터셋으로 return

    2. TestDataset
        - submission을 제출할 때 test 이미지를 뽑기 위한 데이터셋

    3. TrainDatasetForAge
        - 모델을 세가지로 나눠서 학습해보기 위해 나이만 라벨링해주는 데이터셋
    
    4. TrainDatasetForGender
        - 모델을 세가지로 나눠서 학습해보기 위해 성별만 라벨링해주는 데이터셋

    5. TrainDatasetForMask
        - 모델을 세가지로 나눠서 학습해보기 위해 마스크 착용 여부만 라벨링해주는 데이터셋

    - 이외에 test_csv 등 training 및 inference에 사용되는 path를 지정해둔 곳


### loss.py
- loss 클래스를 모아둔 component

    1. Focal Loss
    2. Label Smoothing Loss
    3. F1 Loss

    - 데일리 미션 reference 및 제공된 베이스라인 코드 참고


### model_functions.py
- model에 관련된 함수를 모아둔 component

    1. save_models(model, save_path)
        - training을 진행할 때, 성능이 좋은 경우 저장해두기 위함

    2. func_eval(model, data_iter)
        - training 중 1에폭마다 정확도가 얼마나 나오는지 체크하기 위함

    3. load_models(model_name, save_path, pth_name)
        - 재학습 및 앙상블을 위해 저장해두었던 모델 불러오기 위함


### train_one.py
- 단일 모델로 학습하는 경우(클래스 18개 한번에) 사용하는 component


### train_three.py
- 한 번에 모든 클래스를 분류하지 않고, 나이 성별 마스크 착용 여부를 각각 나눠서 학습하는 경우 (클래스 3, 2, 2) 사용하는 component


### make_submission.py
- 저장한 모델을 불러와서 리더보드를 갱신할 submission.csv를 만들기 위한 component

    1. make_prediction (model_su, model_list, csv_file, data_loader)
        - 단일 모델인 경우와 멀티 헤드인 경우가 있어서 두가지로 나누어주었음


### submission_check.py
- submission 제출 전에 value 분포를 확인하기 위한 만든 component
