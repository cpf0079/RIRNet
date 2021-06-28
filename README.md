# RIRNet
Source codes for paper "[RIRNet: Recurrent-In-Recurrent Network for Video Quality Assessment](https://dl.acm.org/doi/abs/10.1145/3394171.3413717)" in Proceedings of the 28th ACM International Conference on Multimedia (ACM MM ’20).

![image](https://github.com/cpf0079/RIRNet/blob/main/framework.png)

## Usages
### Testing a single sample
Predicting video quality with our model trained on the KoNViD-1k Dataset (coming later).
```
python ./released/demo.py
```
You will get a quality score ranging from 0-5, and a higher value indicates better percerptual quality.

### Training on VQA databases
Reading mos values from the .csv files:
```
python ./released/get_label.py
```
Processing .mp4 files to frames:
```
python ./released/get_frame.py
```
Training the model with 'label_path' and 'frame_path':
```
python ./released/source.py
```

## Environment
* Python 3.6.5
* Pytorch 1.0.1
* Cuda 9.0 Cudnn 7.1 

## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{chen2020RIRNet,
  title={RIRNet: Recurrent-in-Recurrent Network for Video Quality Assessment},
  author={Chen, Pengfei and Li, Leida and Ma, Lei and Wu, Jinjian and Shi, Guangming},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={834--842},
  year={2020}
}
```
