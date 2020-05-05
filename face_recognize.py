import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1

# 使用cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class face_rec():
    def __init__(self):
        # 创建mtcnn对象用于检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5, 0.8, 0.9]

        # 载入facenet用于将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        # model.summary() #模型summary信息
        model_path = './model_data/facenet_keras.h5' # 模型文件
        self.facenet_model.load_weights(model_path)

        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #   face_dataset文件中的图片为人脸数据库
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset") # 列出文件夹中的人脸图片
        self.known_face_encodings = [] # 存储编码后的人脸
        self.known_face_names = [] # 人脸的名字及图片的名字

        # 遍历人脸
        for face in face_list:
            # 图片文件名
            name = face.split(".")[0] 
            # 读取图片是BGR格式数据
            img = cv2.imread("./face_dataset/"+face) 
            # 颜色空间转换
            # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式 
            # cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            # 检测图版中的人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # 长方形转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # Resize to 160x160
            crop_img = cv2.resize(crop_img, (160, 160))
            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)

            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
            print(face_encoding)
            # 记录128维护特征向量到数组
            self.known_face_encodings.append(face_encoding)
            # 记录文件名到数组
            self.known_face_names.append(name)

    def recognize(self, draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        # 得到尺寸
        height, width, _ = np.shape(draw)
        # 颜色空间转换
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 检测出人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles) == 0:
            # 没有找到人脸就退出
            return

        # 长方形转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        # 用于记录检测出来的人脸
        face_encodings = []
        # 遍历人脸
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # resize成160x160
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)
            # 提出128维特征向量
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
            # 记录特征向量
            face_encodings.append(face_encoding)
        # 对应的人名
        face_names = []
        # 遍历刚提取出的特征
        for face_encoding in face_encodings:
            # 取出一张脸特征并与数据库中所有的人脸特征进行对比并计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # 取出文件名
                name = self.known_face_names[best_match_index]
            # 记录到数组中
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            # 画框
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # 写文件名
            cv2.putText(draw, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw


if __name__ == "__main__":
    # 初始化
    dududu = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, draw = video_capture.read()
        dududu.recognize(draw)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
