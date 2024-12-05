from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

project_path = '/mnt/nas4/nto/autotrain'


with DAG('train_pipeline', default_args=default_args, schedule_interval=None) as dag:
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command=(
            f'cd {project_path} && '
            'python make_dataset2.py '
            '-d {{ params.data_dir }} '
            '-ext {{ params.ext }} '
            '-n {{ params.n }} '
            '-sc {{ params.sc }} '
            '-c {{ params.c }} '
            '-tp {{ params.tp }}'
        ),
        params={
            'data_dir': 'data/front',
            'ext': 'png',
            'n': 2,
            'sc': 'body',
            'c': 'front',
            'tp': 0.7,
        }
    )

    make_experiments = BashOperator(
        task_id='make_experiments',
        bash_command=(
            f'cd {project_path} && '
            'python make_experiments.py '
            '--num-keypoints {{ params.num_keypoints }} '
            '--flip-fairs "{{ params.flip_fairs }}" '
            '--data-format {{ params.data_format }} '
            '--data-root {{ params.data_root }} '
            '--begin-epoch {{ params.begin_epoch }} '
            '--end-epoch {{ params.end_epoch }} '
            '--config-file {{ params.config_file }} '
            '--output-file {{ params.output_file }}'
        ),
        params={
            'num_keypoints': 22,
            'flip_fairs': '[[1,2],[3,5],[4,6],[7,8],[9,10],[10,11],[11,12],[13,14],[15,18],[16,17],[19,21],[20,22]]',
            'data_format': 'png',
            'data_root': 'data/front/',
            'begin_epoch': 0,
            'end_epoch': 100,
            'config_file': 'experiments/Auto/hrnet/w48_384x288_adam_lr1e-3.yaml',
            'output_file': 'experiments/Auto/hrnet/w48_384x288_adam_lr1e-3_auto.yaml'
        }
    )

    train = BashOperator(
        task_id='train',
        bash_command=(
            f'cd {project_path} && '
            'python train/main/train.py '
            '--cfg {{ params.cfg }} '
            'MODEL.PRETRAINED {{ params.pretrained_model }}'
        ),
        params={
            'cfg': 'experiments/Auto/hrnet/w48_384x288_adam_lr1e-3_auto.yaml',
            'pretrained_model': 'models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth'
        }
    )
    preprocess >> make_experiments >> train
