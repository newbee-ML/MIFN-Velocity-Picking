# lr=1e-3 BS=32 
python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\hade --DataSet hade --GatherLen 15 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 32 --RepeatTime 0 --lrStart 0.01 --Predthre 0.1 --SGSMode mute 

python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\dq8 --DataSet dq8 --GatherLen 15 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 32 --RepeatTime 0 --lrStart 0.01 --Predthre 0.1 --SGSMode mute

python train.py --OutputPath F:\VSP-MIFN\0Ablation --DataSetRoot E:\Spectrum\dq8 --DataSet dq8 --GatherLen 15 --GPUNO 0 --SizeH 256 --SizeW 256 --SeedRate 0.8 --trainBS 32 --RepeatTime 0 --lrStart 0.01 --Predthre 0.1 --SGSMode mute