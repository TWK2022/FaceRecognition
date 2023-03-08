## 快速人脸特征数据库并进行能识别遮挡的人脸检测和身份识别
>基于insightface官方项目改编:https://github.com/deepinsight/insightface  
>往期项目:https://github.com/TWK2022/insightface  
>首次运行代码时会自动下载模型文件到用户下的.insightface文件夹中  
>在insightface基础上增加了一个识别遮挡概率的分类网络，可以动态的调整识别阈值  
>比如一个人在没遮挡情况下识别概率为0.6，在戴口罩情况下概率大约为0.3，加入的分类网络用于判断遮挡概率  
### 1，database_prepare.py
>将人脸数据库图片放入文件夹image_database中，最好是证件照  
>运行database_prepare.py即可生成人脸特征数据库feature_database.csv  
### 2，predict.py
>将使用电脑视像头，实时预测视像头中截取的画面并显示