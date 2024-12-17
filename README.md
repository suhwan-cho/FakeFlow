# FakeFlow

This is the official PyTorch implementation of our paper:

> **Improving Unsupervised Video Object Segmentation via Fake Flow Generation**, *arXiv 2024*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, DongHyeong Kim, Seunghoon Lee, Sungmin Woo, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/pdf/2407.11714)


<img src="https://github.com/user-attachments/assets/dcbd818a-fff9-417f-a2cf-631df4c82df8" width=800>


You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In unsupervised VOS, the scarcity of training data has been a significant bottleneck in achieving high segmentation accuracy. Inspired by 
observations on two-stream approaches, we introduce a novel data generation method based on the **depth-to-flow conversion** process. With our fake flow generation protocol,
large-scale image-flow pairs can be leveraged during network training. To facilitate future research, we also prepare the **DUTSv2** dataset, which is an extended version of DUTS, 
comprising pairs of the original images and the simulated flow maps.


## Preparation
1\. Download 
[DUTS](http://saliencydetection.net/duts/#org3aad434), 
[DAVIS](https://davischallenge.org/davis2017/code.html),
[FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets),
[YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects),
and [Long-Videos](https://www.kaggle.com/datasets/gvclsu/long-videos)
from the official websites.

2\. Estimate and save optical flow maps from the videos using [RAFT](https://github.com/princeton-vl/RAFT).

3\. For DUTS, simulate optical flow maps using [DPT](https://github.com/isl-org/DPT).

4\. For convenience, I also provide the pre-processed
[DUTSv2](https://drive.google.com/file/d/1P8_USG8CWlpWm5UEcfXgXdr3IYQnhAvi/view?usp=drive_link), 
[DAVIS](https://drive.google.com/file/d/1kx-Cs5qQU99dszJQJOGKNb-wD_090q6c/view?usp=drive_link), 
[FBMS](https://drive.google.com/file/d/1Zgt5ouwFeTpMTemfNeEFz7uEUo77e2ml/view?usp=drive_link), 
[YouTube-Objects](https://drive.google.com/file/d/1t_eeHXJ30TWBNmMzE7vfS0izEafiBfgn/view?usp=drive_link),
and [Long-Videos](https://drive.google.com/file/d/1gZm1QBT_6JmHhphNrxuSztcqkm_eI6Sq/view?usp=drive_link).

5\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Open the "run.py" file.

2\. Specify the model version.

3\. Verify the training settings.

4\. Start **FakeFlow** training!
```
python run.py --train
```


## Testing
1\. Open the "run.py" file.

2\. Specify the model version.

3\. Choose a pre-trained model.

4\. Start **FakeFlow** testing!
```
python run.py --test
```


## Attachments
[pre-trained model (mitb0)](https://drive.google.com/file/d/1FFz9buCu5XCl1LUpwwIUZJNvcAH4V_QG/view?usp=drive_link)\
[pre-trained model (mitb1)](https://drive.google.com/file/d/1DhNsNoF2borozWU5JbrJHIixUoW4hs-Y/view?usp=drive_link)\
[pre-trained model (mitb2)](https://drive.google.com/file/d/1GYAlCt97kcNjtcXoZUGkhIQXgovwV0fz/view?usp=drive_link)\
[pre-computed results](https://drive.google.com/file/d/1OiIaVPf51kqAzGYqFtFl8lLsw-yuY5fi/view?usp=sharing)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```
