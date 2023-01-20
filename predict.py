import cv2
import argparse
import tensorrt
import insightface
import numpy as np
import pandas as pd
import albumentations
import pycuda.autoinit
import pycuda.driver as cuda

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='insightface')
parser.add_argument('--image_root', default='image_predict', type=str, help='|要预测的图片文件夹位置|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置(.csv)|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--threshold', default=0.5, type=float, help='|概率大于阈值判断有此人|')
parser.add_argument('--device', default='cuda', type=str, help='|使用的设备cpu/cuda|')
parser.add_argument('--float16', default=False, type=bool, help='|要与特征数据库精度一致，True为float16，False为float32|')
parser.add_argument('--camera_time', default=20, type=int, help='|预测间隙，单位毫秒，越短显示越不卡顿但越耗性能|')
parser.add_argument('--rgb_mean', default=(0.406, 0.456, 0.485), type=tuple, help='|图片预处理时RGB通道减去的均值|')
parser.add_argument('--rgb_std', default=(0.225, 0.224, 0.229), type=tuple, help='|图片预处理时RGB通道除以的方差|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def draw(image, bbox, name, color):  # 画人脸框
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
    cv2.putText(image, name, (x_min + 5, y_min + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def predict_camera():
    # 加载模型1
    model1 = insightface.app.FaceAnalysis(name='buffalo_l')  # 加载模型，首次运行时会自动下载模型文件到用户下的.insightface文件夹中
    model1.prepare(ctx_id=-1 if args.device == 'cpu' else 0, det_size=(args.input_size, args.input_size))  # 模型设置
    # # 加载模型2
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)  # 创建日志记录信息
    with tensorrt.Runtime(logger) as runtime, open('best.trt', "rb") as f:
        model2 = runtime.deserialize_cuda_engine(f.read())  # 读取模型并构建一个对象
    np_type = tensorrt.nptype(model2.get_tensor_dtype('input'))
    h_input = np.zeros(tensorrt.volume(model2.get_tensor_shape('input')), dtype=np_type)
    h_output = np.zeros(tensorrt.volume(model2.get_tensor_shape('output')), dtype=np_type)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    context = model2.create_execution_context()
    bindings = [int(d_input), int(d_output)]
    # 模型2输入图片的形状转换
    transform = albumentations.Compose([
        albumentations.LongestMaxSize(160),
        albumentations.Normalize(max_pixel_value=255, mean=args.rgb_mean, std=args.rgb_std),
        albumentations.PadIfNeeded(min_height=160, min_width=160,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))])
    # 加载数据库
    df_database = pd.read_csv(args.database_path, dtype=np.float16 if args.float16 else np.float32)
    column = df_database.columns
    feature = df_database.values
    # 打开摄像头
    capture = cv2.VideoCapture(0)
    assert capture.isOpened(), '摄像头打开失败'
    cv2.namedWindow('predict')
    print('|已打开摄像头|')
    # 开始预测
    while capture.isOpened():
        _, image = capture.read()  # 读取摄像头的一帧画面
        pred = model1.get(image)
        if pred != []:
            pred_feature = []  # 记录所有预测的人脸特征
            pred_bbox = []  # 记录所有预测的人脸框
            cover = []  # 记录所有预测的人脸遮挡概率
            for j in range(len(pred)):  # 一张图片可能不只一个人脸
                pred_feature.append(pred[j].normed_embedding)
                bbox = pred[j].bbox
                pred_bbox.append(bbox)
                face_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
                face_image = transform(image=face_image)['image'].transpose(2, 0, 1).reshape(-1).astype(np.float16)
                # 用模型2预测
                cuda.memcpy_htod_async(d_input, face_image, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                cover.append(h_output[0])
            pred_feature = np.array(pred_feature, dtype=np.float16 if args.float16 else np.float32)
            result = np.dot(pred_feature, feature)  # 进行匹配
            for j in range(len(result)):  # 一张图片可能不只一个人脸
                feature_argmax = np.argmax(result[j])
                threshold = args.threshold - 0.2 * cover[j]
                if result[j][feature_argmax] > threshold:
                    name = column[feature_argmax] + '_{:.2f}_{:.2f}'.format(result[j][feature_argmax], cover[j])
                    color = (0, 255, 0)  # 绿色
                else:
                    name = 'None_{:.2f}_{:.2f}'.format(result[j][feature_argmax], cover[j])
                    color = (0, 0, 255)  # 红色
                # 画人脸框
                image = draw(image, pred_bbox[j], name, color)
        cv2.imshow('predict', image)
        cv2.waitKey(max(args.camera_time, 1))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_camera()
