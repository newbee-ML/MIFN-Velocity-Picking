# MIFN

## 0 准备工作

### 0.1 配置Python、CUDA与Pytorch

* Python 3.7

* CUDA 11.0

* Pytorch  1.7.1  GPU

  ```shell
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  ```

### 0.2 配置Python所需环境

* 根据requirements.txt配置Python所需第三方库

```shell
pip install -r requirements.txt
```

### 0.3 数据文件要求

* 必须按照如下文件结构放置数据

  --- Root

  ​        |___   segy

  ​					 |___  vel.stk.sgy 

  ​					 |___  vel.pwr.sgy 

  ​					 |___  vel.gth.sgy

  ​		|___   t_v_labels.dat

## 1 代码结构

```
.
├── loss  # 损失函数
│   ├── detail_loss.py
│   ├── __init__.py
│   ├── loss.py
│   └── util.py
├── net  # 保存网络的基本结构
│   ├── BasicModule.py
│   ├── __init__.py
│   └── MIFNet.py
├── utils  # 工具函数
│   ├── BuiltStkDataSet.py
│   ├── evaluate.py
│   ├── __init__.py
│   ├── LabTxt2Npy.py
│   ├── LoadData.py
│   ├── logger.py
│   ├── LogWriter.py
│   ├── metrics.py
│   ├── PastProcess.py
│   ├── PlotTools.py
│   └── SpecEnhanced.py
└── predict.py  # 预测主文件
└── train.py    # 训练主文件
```

## 2 训练模型

以下代码请在终端运行，当前目录为代码的根目录。

### 2.1 生成索引数据集

在数据文件的根目录下生成数据样本索引：

```shell
# Transfer label file "t_v_labels.dat" to “t_v_labels.npy”
python utils/LabTxt2Npy.py /Root/t_v_labels.dat /Root/t_v_labels.npy
# Make H5 dataset
python utils/BuiltStkDataSet.py /Root
```

### 2.2 训练新数据集的模型

训练文件中可变参数说明：

| 参数名      | 类型  | 默认值 | 说明                                             |
| ----------- | ----- | ------ | ------------------------------------------------ |
| DataSetRoot | str   | 无     | 数据集根目录地址                                 |
| DataSet     | str   | test   | 训练集名称（用于记录训练结果，可任意命名）       |
| GatherLen   | int   | 21     | 单个叠加段最大宽度，用于编码，注意必须是最大宽度 |
| SGSMode     | str   | all    | 叠加段道集编码方式 all (正负分离并叠加), mute(正常标准化)|
| SeedRate    | float | 0.5    | 训练集占数据集的比例                             |
| ReTrain     | int   | 1      | 是否重新训练                                     |
| GPUNO       | int   | 0      | 训练所用GPU编号                                  |
| SizeH       | int   | 256    | 输入网络图像高度，必须为32倍数                   |
| Predthre    | float | 0.3    | 网络预测图的分类阈值                             |
| lrStart     | float | 0.01   | 初始学习率                                       |
| trainBS     | int   | 16     | 训练的batchsize                                  |
| valBS       | int   | 16     | 验证的batchsize                                  |

```shell
# 激活Python环境
conda activate env_name
# 训练新模型 其他参数也可变，可查询上述参数表做相应更改 “--”后为参数名，再后接空格加自定义参数如“--lrStart 0.02”
python train.py --DataSetRoot Root --DataSet SetName --GatherLen GatherLen --SeedRate 0.6 --trainBS 16

# 例子
python train.py --DataSetRoot /home/colin/data/Spectrum/hade --DataSet hade --GatherLen 15 --SeedRate 0.6 --trainBS 16
```

注：训练过程，可以通过Tensorboard查看

## 3 预测模型

测试文件中可变参数说明：

| 参数名      | 类型 | 默认值 | 说明                                                         |
| ----------- | ---- | ------ | ------------------------------------------------------------ |
| LoadModel   | str  | 无     | 模型地址（“.pth”的绝对地址）                                 |
| DataSetRoot | str  | 无     | 数据集根目录地址                                             |
| DataSet     | str  | test   | 训练集名称（用于记录训练结果，可任意命名）                   |
| GatherLen   | int  | 21     | 单个叠加段最大宽度，用于编码，注意必须是最大宽度（保持与训练模型相同） |
| Resize      | int  | 1      | 是否重新保存测试                                             |
| GPUNO       | int  | 0      | 训练所用GPU编号                                              |
| SizeH       | int  | 256    | 输入网络图像高度，必须为32倍数（保持与训练模型相同）         |
| PredBS      | int  | 32     | 预测的batchsize                                              |

```shell
# 激活Python环境
conda activate env_name
# 训练新模型 其他参数也可变，可查询上述参数表做相应更改 “--”后为参数名，再后接空格加自定义参数如“--lrStart 0.02”
python predict.py --LoadModel PthPath --DataSetRoot Root --DataSet SetName --GatherLen GatherLen --PredBS 16

# 例子
python predict.py --LoadModel /home/colin/Project/spectrum/MIFN-Submit/result/hade/model/hade_256_0.6.pth --DataSetRoot /home/colin/data/Spectrum/hade --DataSet hade --GatherLen 15 --PredBS 16
```

注：结果保存在CSV文件中，共四列，分别为line, trace, time, velocity
