# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/aiops_ygq/tianchi_aiops:v1

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 安装所需要的包
# RUN pip3 install -r requirements.txt
# pip install pandas 
# pip install gensim
# pip install numpy
# pip install scikit-learn
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# pip install torchtext
RUN apt -y update
RUN apt install zip

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]

