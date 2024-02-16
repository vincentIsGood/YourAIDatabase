sudo docker build -t aidb:1.0 -f ./YourAIDatabase/docker/Dockerfile_cpu .

## RUN
sudo docker run -p 5022:5022 -p 5023:5023 \ 
    -v $PWD/YourAIDatabase/chroma_db:/app/YourAIDatabase/chroma_db \
    -v $PWD/YourAIDatabase/docs:/app/YourAIDatabase/docs \
    -v $PWD/YourAIDatabase/models:/app/YourAIDatabase/models \
    -v $PWD/YourAIDatabase/configs:/app/YourAIDatabase/configs \
    -it <container_id> ./start.sh