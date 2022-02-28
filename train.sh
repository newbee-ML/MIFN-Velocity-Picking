## 2.28 valid the input size 
python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\hade --DataSet hade --GatherLen 11 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 8 --RepeatTime 0 --lrStart 0.001 --Predthre 0.1 --SGSMode mute --ReTrain 0

python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\hade --DataSet hade --GatherLen 11 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 16 --RepeatTime 0 --lrStart 0.001 --Predthre 0.1 --SGSMode mute

python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\hade --DataSet hade --GatherLen 11 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 32 --RepeatTime 0 --lrStart 0.001 --Predthre 0.1 --SGSMode mute