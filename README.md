## 📚 프로젝트 소개
본 레포지토리는 DevOps 및 인프라 자동화를 위한 주요 명령어 및 스크립트를 모아 관리하는 **Command Pack**입니다.  
반복적인 명령어 입력을 줄이고, 업무 효율을 높이기 위해 다양한 기술 스택별로 필요한 명령어 모음 파일을 제공합니다.

## 🚀 주요 기능
- Docker, Kubernetes, Linux, Git 등 주요 스택의 명령어 자동완성 스크립트 제공
- DevOps 실습 및 실무 환경에서 명령어 입력 시간을 단축
- 개인 학습 및 업무 자동화 도구로 활용 가능

## 🛠️ 폴더 구조

util
command_pack
 └──util
    ├── docker.sh
    └── kubernetes.sh


## 📦 설치 및 사용 방법
```bash
git clone https://github.com/tkdals69/util
cd util/* /etc/bash_completion.d/
source /etc/bash_completion.d/docker
source /etc/bash_completion.d/kubectl


## 방법 1. /etc/profile.d/bash_completion.sh 마지막에 수동 로딩 추가
if [ -d /etc/bash_completion.d ]; then
    for bcfile in /etc/bash_completion.d/*; do
        [ -f "$bcfile" ] && . "$bcfile"
    done
fi

