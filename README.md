# FakeFlow

This is the official PyTorch implementation of our paper:

> **Improving Unsupervised Video Object Segmentation via Fake Flow Generation**, *arXiv 2024*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, DongHyeong Kim, Seunghoon Lee, Sungmin Woo, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/pdf/2407.11714)


![image](https://github.com/user-attachments/assets/a30b7cac-90cc-4ef0-96e8-927d0b6c23f4)


You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In unsupervised VOS, the scarcity of training data has been a significant bottleneck in achieving high segmentation accuracy. Inspired by 
observations on two-stream approaches, we introduce a novel data generation method based on the **depth-to-flow conversion** process. With our fake flow generation protocol,
large-scale image-flow pairs can be leveraged during network training. To facilitate future research, we also prepare the **DUTSv2** dataset, which is an extended version of DUTS, 
comprising pairs of the original images and the simulated flow maps.


## Preparation
1\. Download [DUTS](http://saliencydetection.net/duts/#org3aad434) and [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data) for network training.

2\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) for network training and testing.

3\. Download [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets) for network testing.

4\. Download [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects) for network testing.

5\. Download [Long-Videos](https://www.kaggle.com/datasets/gvclsu/long-videos) for network testing.

6\. Save optical flow maps of YouTube-VOS 2018, DAVIS, FBMS, YouTube-Objects, and Long-Videos using [RAFT](https://github.com/princeton-vl/RAFT).

7\. For convenience, I also provide the pre-processed [DUTSv2](https://drive.google.com/file/d/1P8_USG8CWlpWm5UEcfXgXdr3IYQnhAvi/view?usp=drive_link), [YouTube-VOS](https://drive.google.com/file/d/1Lw7pN0z4JwF9nr1eku33hvynkBpbiiQt/view?usp=drive_link), 
[DAVIS](https://drive.google.com/file/d/1kx-Cs5qQU99dszJQJOGKNb-wD_090q6c/view?usp=sharing), [FBMS](https://drive.google.com/file/d/1Zgt5ouwFeTpMTemfNeEFz7uEUo77e2ml/view?usp=sharing), [YouTube-Objects](https://drive.google.com/file/d/1t_eeHXJ30TWBNmMzE7vfS0izEafiBfgn/view?usp=sharing), and [Long-Videos](https://drive.google.com/file/d/1gZm1QBT_6JmHhphNrxuSztcqkm_eI6Sq/view?usp=sharing).

8\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Move to "run.py" file.

2\. Define model version (ver): 'mitb0' or 'mitb1' or 'mitb2'

3\. Check training settings.

4\. Run **FakeFlow** training!!
```
python run.py --train
```


## Testing
1\. Move to "run.py" file.

2\. Define model version (ver): 'mitb0' or 'mitb1' or 'mitb2'

3\. Select a pre-trained model that accords with the defined model version.

4\. Run **FakeFlow** testing!!
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
