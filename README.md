# mrc_flask_api

### 1. build base iamges
    cd base_images
    docker build --tag python-base-image:3.7.13 .

### 2. docker compose up
    docker-compose up --build -d