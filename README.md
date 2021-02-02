

# 机器学习作业-2 动作识别

|文件名|作用|
|:--:|:--:|
|preprocess.py	|预处理文件，在训练前先运行。提高训练速度|
|train_model.py	|主函数 python train_model.py|
|test_model.py	|测试函数 接受.avi视频输入，预测标签|
|dataloader	|加载数据集|
|decoder	|LSTM模型|
|encoder	|特征提取部分网络|


说明：

+ 运行时会下载Pytorch的r(2+1)d模型文件和Faster-RCNN文件，他们是预训练的模型，用于提取下游任务的特征。
+ 用该代码进行训练，请先运行preprocess.py文件提取数据集特征，这是为了加速训练过程，不用每个epoch都提取重复的特征。推理时不需进行预处理。
+ Test_model提供了使用avi文件进行单独预测的接口。需要修改video_dir = '../dataset/val_data'的路径。该文件将对目录下所有avi文件进行推理。
+ 默认使用cuda:1，可以修改为其他。
+ 需要安装的包：cv2, pytorch, torchvision
