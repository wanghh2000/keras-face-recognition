## Face-Recognition：人脸识别算法在Keras当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [使用方法 Usage](#使用方法)
4. [效果 Performance](#Reference)

### 所需环境
conda install python==3.6.12
pip install tensorflow-gpu==1.15.4  
pip install keras==2.2.4
conda install opencv==3.4.2
pip uninstall scipy & pip install scipy=1.5.2
pip uninstall h5py  & pip install h5py
pip install matplotlib

### 文件下载
训练所需的facenet_keras.h5可以在Release里面下载。  
也可以去百度网盘下载  
链接: https://pan.baidu.com/s/1A9jCJa_sQ4D3ejelgXX2RQ 提取码: tkhg  
### 使用方法
1、先将整个仓库download下来。  
2、下载完之后解压，同时下载facenet_keras.h5文件。  
3、将facenet_keras.h5放入model_data中。  
4、将自己想要识别的人脸放入到face_dataset中。  
5、运行face_recognize.py即可。  
6、align.py可以查看人脸对齐的效果。  
### 效果
face_recognize.py的运行结果：  
![result](/result/result.png))  
