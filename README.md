# Airflow Keypoint Pipeline

Airflow를 사용하여 Keypoint detection 파이프라인을 자동화하는 프로젝트입니다. 이 파이프라인은 데이터셋 생성, 실험 설정, 모델 훈련 단계로 구성되어 있습니다.

## 프로젝트 구조

- `train_pipeline.py`: 전체 파이프라인을 정의하는 Airflow DAG 파일
- `make_dataset2.py`: 데이터셋을 생성하는 파일
- `make_experiments.py`: 실험 설정 파일을 업데이트하는 파일
- `train/main/train.py`: 모델을 훈련하는 파일


## 사용법

### 1. 데이터셋 생성

`make_dataset2.py` 파일은 **labelme** tool로 라벨링을 진행한 JSON 파일과 이미지 쌍 데이터를 COCO 스타일의 데이터셋을 생성

#### 사용법
bash
python make_dataset2.py -d <data_dir> -ext <img_ext> -n <num_points> -sc <supercategory> -c <category> -tp <train_per>
- `-d`, `--data_dir`: 데이터 디렉토리의 절대 경로
- `-ext`, `--img_ext`: 이미지 파일 확장자 (예: jpg, png)
- `-n`, `--num_points`: 키포인트의 수
- `-sc`, `--supercategory`: 슈퍼카테고리
- `-c`, `--category`: 카테고리
- `-tp`, `--train_per`: 훈련 데이터 비율 (예: 0.8)



### 2. 실험 설정

`make_experiments.py` 스크립트는 실험 설정 파일을 업데이트합니다.

#### 사용법
bash
python make_experiments.py --num-keypoints <num_keypoints> --flip-fairs <flip_fairs> --data-format <data_format> --data-root <data_root> --begin-epoch <begin_epoch> --end-epoch <end_epoch> --config-file <config_file> --output-file <output_file>


- `--num-keypoints`: 키포인트의 수
- `--flip-fairs`: 플립 페어 옵션 (리스트 형태의 문자열)
- `--data-format`: 데이터 형식
- `--data-root`: 데이터 루트 경로
- `--begin-epoch`: 시작 에포크
- `--end-epoch`: 종료 에포크
- `--config-file`: 원본 설정 파일 경로
- `--output-file`: 업데이트된 설정 파일 저장 경로

### 3. 모델 훈련

`train/main/train.py` 스크립트는 모델을 훈련합니다.

#### 사용법
bash
python train/main/train.py --cfg <config_file> MODEL.PRETRAINED <pretrained_model>

- `--cfg`: 실험 설정 파일 경로
- `MODEL.PRETRAINED`: 사전 훈련된 모델 경로

### 4. Airflow 파이프라인 실행

`train_pipeline.py` 파일은 Airflow DAG을 정의합니다. 이 파일을 통해 전체 파이프라인을 실행할 수 있습니다.

#### 사용법

Airflow 웹 UI 또는 CLI를 통해 DAG을 실행합니다.
