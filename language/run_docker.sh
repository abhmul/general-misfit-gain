sudo docker build . -t pagi:latest
sudo docker run --rm --gpus all --network host -v $(pwd)/..:/code -w /code --env-file ./env.list -it pagi:latest bash
