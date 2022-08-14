# mrc_flask_api


### 1. build base iamges
    cd base_images
    docker build --tag python-base-image:3.7.13 .

### 2. docker compose up - 로그 로컬 저장
    docker-compose -f mrc_api.yml up --build -d


### 2-1. docker compose up - logstash로 전송
    docker-compose -f mrc_api_logstash.yml up --build -d
