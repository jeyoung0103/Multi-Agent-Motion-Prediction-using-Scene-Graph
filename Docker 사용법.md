
# Docker 이미지 빌드 매뉴얼 (QCNet 학습 환경)

본 매뉴얼은 Ubuntu 환경에서 미래 궤적 예측 모델 학습을 위한 Docker 이미지를 직접 빌드하는 절차를 설명합니다.

## 1.Docker 설치
 
Ubuntu 환경에서 아래 명령어를 순차적으로 실행하여 Docker를 설치합니다:

```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
docker --version 
``` 


## 2.Docker 이미지 빌드
 
Docker 파일 'Dockerfile_qcnet_torch2.1.0' 을 이용하여 이미지를 빌드 합니다:

```
docker build -t qcnet-env:torch2.1.0 -f Dockerfile_qcnet_torch2.1.0 .
``` 

* -t : 생성될 이미지의 이름 및 태그
* -f : 사용할 Dockerfile 이름
* . : 현재 디렉토리를 build context로 사용

## 3.Docker 이미지 실행
 
생성된 Docker 이미지를 실행합니다:

```
docker run --gpus all -it --name ETRI --ipc=host --shm-size=8g -v /mnt/hdd1/jeyoung/kadif:/workspace qcnet-env:torch2.1.0 /bin/bash
``` 
# 종료된 컨테이너 다시 시작
docker start ETRI
# 컨테이너 접속
docker exec -it ETRI /bin/bash

* /path/to/QCNet에는 host가 실행중인 Docker와 공유할 폴더 경로 입력
* 공유 폴더는 /workspace로 Docker 컨테이너 안에서 mount됨

## 4.Docker 이미지 저장 및 공유

이미지 저장
``` 
docker save -o qcnet-env:torch2.1.0.tar qcnet-env:torch2.1.0
``` 

다른 시스템에서 불러오기
``` 
docker load -i qcnet-env:torch2.1.0.tar
``` 

## 5. (Optional) NVidia Docker Container 설치

* Install NVIDIA Container Toolkit

```
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

sudo apt update
sudo apt install -y curl gnupg lsb-release

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit
```
* Configure Docker to use NVIDIA runtime

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Docker 명령어 모음


* 동작중인 컨테이너 확인

```bash
$ docker ps
```

* 정지된 컨테이너 확인

```bash
$ docker ps -a
```

* 컨테이너 삭제

```bash
$ docker rm [컨테이너Id]
```

* 삭제된 것 확인

```bash
$ docker ps -a
```

* 복수개 삭제도 가능

```bash
$ docker rm [컨테이너Id], [컨테이너Id]
```

* 컨테이너 모두 삭제

```bash
$ docker rm `docker ps -a -q`
```

* 현재 이미지 확인

```bash
$ docker images
```

* 이미지 삭제

```bash
$ docker rmi [이미지Id]
```

* 컨테이너를 삭제하기 전에 이미지를 삭제할 경우

> `-f` 옵션을 붙이면 컨테이너도 강제삭제

```bash
$ docker rmi -f [이미지Id]
```


