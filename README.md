# Airflow Keypoint Pipeline
![스크린샷 2024-11-21 오후 3 58 15](https://github.com/user-attachments/assets/37f79603-4e90-418f-b81c-190220e29221)

Airflow와 Mlflow를 사용하여 Keypoint detection 자동화 파이프라인 모델 관리를 구현한 프로젝트입니다. 파이프라인은 데이터셋 생성, 실험 설정, 모델 훈련 단계로 구성되어 있습니다.

## 프로젝트 구조

- `train_pipeline.py`: 전체 파이프라인을 정의하는 Airflow DAG 파일
- `make_dataset2.py`: 데이터셋을 생성하는 파일
- `make_experiments.py`: 실험 설정 파일을 업데이트하는 파일
- `train/main/train.py`: 모델을 훈련하는 파일

## 프로젝트 상황
- Hrnet 기반 Keypoint detection을 transfer-learning하여 다양한 카테고리에 접목 필요
- coco 데이터 형식 설정부터 모델 훈련까지 동일한 작업 반복 -> 자동화 필요
- 원하는 카테고리를 빠르게 학습하여 시연 필요 -> 학습부터 배포까지의 시간 단축 필요
- 많은 사람들이 동시에 모델 학습을 돌리게 되는 상황 발생 -> 모델 버전 관리 및 기록 필요

## 파이프라인

### 1. 데이터셋 생성

`make_dataset2.py` 

파일은 annotation tool인 **labelme** 로 라벨링을 진행한 JSON 파일과 이미지 쌍 데이터를 COCO 스타일의 데이터셋으로 생성
![스크린샷 2024-11-21 오후 3 59 54](https://github.com/user-attachments/assets/6f6d529c-cc3d-495b-b3eb-fa2e6161c76f)

#### 사용법


```bash
python make_dataset2.py -d <data_dir> -ext <img_ext> -n <num_points> -sc <supercategory> -c <category> -tp <train_per>
```


- `-d`, `--data_dir`: 데이터 디렉토리 경로
- `-ext`, `--img_ext`: 이미지 파일 확장자 (예: jpg, png)
- `-n`, `--num_points`: 카테고리 키포인트의 수
- `-sc`, `--supercategory`: 상위 카테고리 (ex : 신체)
- `-c`, `--category`: 카테고리 (ex : 허리)
- `-tp`, `--train_per`: Train Validation 데이터 비율 (예: 0.8)



### 2. 실험 설정

`make_experiments.py`

**학습을 위한 실험 설정 파일을 업데이트**

#### 사용법

```bash
python make_experiments.py --num-keypoints <num_keypoints> --flip-fairs <flip_fairs> --data-format <data_format> --data-root <data_root> --begin-epoch <begin_epoch> --end-epoch <end_epoch> --config-file <config_file> --output-file <output_file>
```

- `--num-keypoints`: 카테고리 키포인트의 수
- `--flip-fairs`: 키포인트 좌우 대칭 리스트 (ex : [[1,2],[3,4]])
- `--data-format`: 데이터 형식
- `--data-root`: 데이터 경로
- `--begin-epoch`: 시작 epoch
- `--end-epoch`: 종료 epoch
- `--config-file`: 기존 설정 파일 경로
- `--output-file`: 업데이트된 설정 파일 저장 경로

### 3. 모델 훈련

`train/main/train.py` 

**모델 훈련 진행**

#### 사용법
```python
python train/main/train.py --cfg <config_file> MODEL.PRETRAINED <pretrained_model>
```

- `--cfg`: 실험 설정 파일 경로
- `MODEL.PRETRAINED`: 사전 훈련된 모델 경로

## Model Registry

### 개요

MLflow를 사용하여 keypoint detection 모델 버전 관리 및 기록

### 구성

1. 실험 시작 
- hyperparameter 기록
    ```python
    with mlflow.start_run():
        # 하이퍼파라미터 기록
        # 필요한 다른 파라미터도 기록 가능
        mlflow.log_param("learning_rate", cfg.TRAIN.LR)
        mlflow.log_param("batch_size", train_batch_size)
    ```
2. metric 기록
    ```python
    mlflow.log_metric("epoch_val_loss", epoch_loss_sum / num_batches, step=epoch)
    mlflow.log_metric("epoch_val_acc", epoch_acc_sum / num_batches, step=epoch)
    ```
    batch 단위로 loss 및 accuracy 평균 기록
    
3. metric 기록

-  가장 높은 AP를 가진 모델 저장
    ```python
    if best_model:
        mlflow.pytorch.log_model(model, "best_model")
    ```
- final model 저장
    ```python
    mlflow.pytorch.log_model(model, "final_model")
    ```
