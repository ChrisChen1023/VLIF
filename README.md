<div align="center">

  
  # [AAAI 2026 ☔] Exploring the Potentials of Spiking Neural Networks for Image Deraining 🎅

</div>

**Abstract:** 
Biologically plausible and energy-efficient frameworks such as Spiking Neural Networks (SNNs) have not been sufficiently explored in low-level vision tasks. Taking image deraining as an example, this study addresses the representation of the inherent high-pass characteristics of spiking neurons, specifically in image deraining and innovatively proposes the Visual LIF (VLIF) neuron, overcoming the obstacle of lacking spatial contextual understanding present in traditional spiking neurons. To tackle the limitation of frequency-domain saturation inherent in conventional spiking neurons, we leverage the proposed VLIF to introduce the Spiking Decomposition and Enhancement Module and the lightweight Spiking Multi-scale Unit for hierarchical multi-scale representation learning. Extensive experiments across five benchmark deraining datasets demonstrate that our approach significantly outperforms state-of-the-art SNN-based deraining methods, achieving this superior performance with only 13% of their energy consumption. These findings establish a solid foundation for deploying SNNs in high-performance, energy-efficient low-level vision tasks.

![vlif_lif4](https://github.com/user-attachments/assets/bbc3b477-8125-4416-84e3-604b6e9bd88d)


## Preparation

## 🔔 Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain12</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
    <th>RW_Rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="https://drive.google.com/drive/folders/15ysVRwdpoP8wyxdf8jKDl09YNS3mAUmA?usp=drive_link">Download</a> </td>
    <td align="center"> <a href="https://drive.google.com/drive/folders/1d-hhRuBvrtYhJrclzyPXeWhzpoVX5WFt?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/1O_EzLBmT9VLGHaqzole27r-xBT5FaeD2?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/1Azd1T7Us1LGALFH19eNI2820yjF7h_RK?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/1BjAlSKNlAdh_MzfSOYNUha0_W6GfmuvI?usp=drive_link">Download</a> </td>
  </tr>
</tbody>
</table>


## 🤖 Pre-trained Models
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="https://drive.google.com/file/d/1IVyYsPkUS43HbYyrwAFIvX-NrB6Bkmaa/view?usp=sharing">Download</a> </td>
    <td align="center"> <a href="https://drive.google.com/file/d/1mkdFWZebdBD9n4Jrmy-yAlYdUDPA_keX/view?usp=drive_link">Download</a> </td>
    <td > <a href="https://drive.google.com/file/d/1pEBkthyTpS-r1G-_KPdMp83XRncFd4A3/view?usp=drive_link">Download</a> </td>
  </tr>
</tbody>
</table>


## Install 

1. Create a new conda environment
```
conda create -n VLIF python=3.8
conda activate VLIF 
```

2. Install dependencies
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib scikit-image opencv-python numpy einops math natsort tqdm lpips time tensorboardX
```

### Download

You can download the pre-trained models and datasets on Google Drive.

The final file path should be the same as the following:

```
┬─ pretrained_models
│   ├─ Rain200L.pth
│   ├─ Rain200H.pth
│   ├─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ Rain200L
    ├─ Rain200H
    │├─ train
    ││  ├─ input
    ││  │  └─ ... (image filename)
    ││  │  └─ target
    ││  │  └─ ... (corresponds to the former)
    │└─ test
    │    └─ ...
    └─ ... (dataset name)

```

### 🛠️ Training & Testing

### Train
You should change the path to yours in the `Train.py` file:
```
--train_dir /..../..../.../train
--val_dir /.../.../.../val
```

Then run the following script to test the trained model:

```sh
python train.py
```

### Test
You should change the path to yours in the `Test.py` file:
```
--data_path /..../..../.../test/input
--target_path /.../.../.../test/target
--weights /.../.../../....pth
```
Then run the following script to test the trained model:

```sh
python test.py
```

## 👍 Acknowledgment

This code is based on the [ESDNet](https://github.com/MingTian99/ESDNet), [spikingjelly](https://github.com/fangwei123456/spikingjelly). Thanks for their awesome work.
