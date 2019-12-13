# AU331 FCCM: 基于非小细胞肺癌CT图像的医疗诊断与病情分析


张弛， 程铭

老师的建议：“你们的内容属于比较标准和热门的研究问题，因此如果只是把数据用自己搭建的模型跑一下结果，工作量不足以支撑大作业的工作。可以考虑调研一下，在他人已有的工作基础上进行一些创新的尝试，如果结果好可以考虑发论文。”

运行指令
```
python3 train.py --dataset_dir /home/zhang/documents/data/curriculum/MLproject/pre-process  \
--label_dir /home/zhang/documents/data/curriculum/MLproject --batch-size 4    \
--epoch-size 1000  --lr 1e-4    --epochs 100  --name deemo
```

可视化窗口
```
tensorboard --logdir=./
```
