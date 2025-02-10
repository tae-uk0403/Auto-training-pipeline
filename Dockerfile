# # ---------- Dockerfile 시작 ----------
# # (1) Python 3.8 slim 이미지를 베이스로 사용
# FROM python:3.8-slim

# # (2) 작업 디렉토리를 /app 으로 설정
# WORKDIR /app

# # (3) 필요한 시스템 패키지 설치 (C/C++ 컴파일러, JPEG 라이브러리, glib 등)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libjpeg-dev \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# # (4) requirements.txt만 먼저 복사
# COPY autotrain-requirements.txt /app/requirements.txt

# # (5) pip 및 선행 패키지 설치
# #     - 여러 pip install 명령을 하나의 RUN으로 묶으면 레이어 수가 줄어듭니다.
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir cython numpy \
#     && pip install --no-cache-dir -r requirements.txt

# # (6) 프로젝트 전체 파일 복사
# COPY . /app

# # (7) (선택) 환경 변수 설정
# # ENV PYTHONPATH /app

# # (8) 컨테이너 실행 시 기본적으로 bash 셸에 들어가도록 설정
# ENTRYPOINT ["/bin/bash"]
# # ---------- Dockerfile 끝 ----------





# ---------- Dockerfile 시작 ----------
# (1) Python 3.8 slim 이미지를 베이스로 사용
FROM python:3.8-slim

# (2) 작업 디렉토리를 /app 으로 설정
WORKDIR /app

# (3) 필요한 시스템 패키지 설치 (C/C++ 컴파일러, JPEG 라이브러리, glib 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# (4) requirements.txt만 먼저 복사
COPY autotrain-requirements.txt /app/requirements.txt

# (5) pip 및 선행 패키지 설치
#     - 여러 pip install 명령을 하나의 RUN으로 묶으면 레이어 수가 줄어듭니다.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir cython numpy \
    && pip install --no-cache-dir -r requirements.txt

# (5.1) pycocotools 내 cocoeval.py 수정
#       (pycocotools가 이미 설치된 후에야 /usr/local/lib/python3.8/site-packages/... 경로가 존재)
# RUN sed -i \
#     's/np.round((0.95 - .5) \/ .05) + 1/int(np.round((0.95 - .5) \/ .05)) + 1/g' \
#     /usr/local/lib/python3.8/site-packages/pycocotools/cocoeval.py \
#  && sed -i \
#     's/np.round((1.00 - .0) \/ .01) + 1/int(np.round((1.00 - .0) \/ .01)) + 1/g' \
#     /usr/local/lib/python3.8/site-packages/pycocotools/cocoeval.py

# (6) 프로젝트 전체 파일 복사
COPY . /app

# (7) (선택) 환경 변수 설정
# ENV PYTHONPATH /app

# (8) 컨테이너 실행 시 기본적으로 bash 셸에 들어가도록 설정
ENTRYPOINT ["/bin/bash"]
# ---------- Dockerfile 끝 ----------
