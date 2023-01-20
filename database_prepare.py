import os
import cv2
import tqdm
import argparse
import insightface
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='insightface')
parser.add_argument('--image_root', default='./image_database', type=str, help='|人脸数据库图片位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--device', default='cuda', type=str, help='|使用的设备cpu/cuda|')
parser.add_argument('--float16', default=False, type=bool, help='|特征数据库精度，True为float16，False为float32|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def database_prepare():
    model = insightface.app.FaceAnalysis(name='buffalo_l')  # 加载模型，首次运行时会自动下载模型文件到用户下的.insightface文件夹中
    model.prepare(ctx_id=-1 if args.device == 'cpu' else 0, det_size=(args.input_size, args.input_size))  # 模型设置
    image_dir = sorted(os.listdir(args.image_root))  # 获取图片路径
    feature = [_ for _ in range(len(image_dir))]  # 记录人脸特征
    column = [_ for _ in range(len(image_dir))]  # 记录特征对应的名称
    for i in tqdm.tqdm(range(len(image_dir))):
        image = cv2.imread(args.image_root + '/' + image_dir[i])
        pred = model.get(image)
        column[i] = image_dir[i].split('.')[0]
        feature[i] = pred[0].normed_embedding
    feature = np.array(feature, dtype=np.float16 if args.float16 else np.float32)
    df_database = pd.DataFrame(feature.T, columns=column)
    df_database.to_csv('feature_database.csv', index=False)  # 将特征保存至csv文件
    print('| 特征数据库准备完毕 图片总量:{} |'.format(len(image_dir)))


if __name__ == '__main__':
    database_prepare()
