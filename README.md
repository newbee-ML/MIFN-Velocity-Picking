# MIFN
---
The code for paper 'Automatic Velocity Picking Using a Multi-Information Fusion Deep Semantic Segmentation Network' (MIFN)

If you cite this paper, please use the following bibtex:
```bibtex
@article{wang2022automatic,
  title={Automatic Velocity Picking Using a Multi-Information Fusion Deep Semantic Segmentation Network},
  author={Wang, Hongtao and Zhang, Jiangshe and Zhao, Zixiang and Zhang, Chunxia and Long, Li and Yang, Zhiyu and Geng, Weifeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--10},
  year={2022},
  publisher={IEEE}
}
```

## 0 Preparing

### 0.1 Python, CUDA and Pytorch

* Python 3.7

* CUDA 11.0

* Pytorch  1.7.1  GPU

  ```shell
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  ```

### 0.2 Python packages

* install packages based on `requirements.txt`

```shell
pip install -r requirements.txt
```

### 0.3 Requirement structure for data folder

* You should follow the structure for repreducing

  --- Root

  ​        |___   segy

  ​					 |___  vel.stk.sgy 

  ​					 |___  vel.pwr.sgy 

  ​					 |___  vel.gth.sgy

  ​		|___   t_v_labels.dat

## 1 code structure

```
MIFN-VELOCITY-PICKING
│  predict.py                 # main predict 
│  README-ch.md               # readme in CH
│  README.md                  # readme in EN
│  requirements.txt           # python packages list
│  test.py                    # main test
│  train.py                   # main train
│
├─loss  // loss function
│      detail_loss.py
│      loss.py
│      util.py
│      __init__.py
│
├─net  // our proposed MIFN
│      AblationNet.py
│      BasicModule.py
│      MIFNet.py
│      __init__.py
│
├─Tuning  // tuning
│      tuning.py
│
└─utils  // other tools
        BuiltStkDataSet.py
        evaluate.py
        GetNMOResult.py
        LabTxt2Npy.py
        LoadData.py
        logger.py
        metrics.py
        PastProcess.py
        PlotTools.py
        remove.py
        SpecEnhanced.py
        __init__.py
```

## 2 train your model

Please run the following code in the repo root.

### 2.1 transfer labels file type and generate h5 datasets 

```shell
# Transfer label file "t_v_labels.dat" to “t_v_labels.npy”
python utils/LabTxt2Npy.py /Root/t_v_labels.dat /Root/t_v_labels.npy
# Make H5 dataset
python utils/BuiltStkDataSet.py /Root
```

### 2.2 training processing

```shell
# run on cmd (windows) or terminal (linux)
python train.py --DataSetRoot Root --DataSet SetName --GatherLen GatherLen --SeedRate 0.6 --trainBS 16
# an example
python train.py --DataSetRoot /home/colin/data/Spectrum/hade --DataSet hade --GatherLen 15 --SeedRate 0.6 --trainBS 16
```


## 3 predict

```shell
# run on cmd (windows) or terminal (linux)
python predict.py --LoadModel PthPath --DataSetRoot Root --DataSet SetName --GatherLen GatherLen --PredBS 16

# an example
python predict.py --LoadModel /home/colin/Project/spectrum/MIFN-Submit/result/hade/model/hade_256_0.6.pth --DataSetRoot /home/colin/data/Spectrum/hade --DataSet hade --GatherLen 15 --PredBS 16
```

tips：results saved in CSV file, including four columns: 1 line, 2 trace, 3 time, 4 velocities.
