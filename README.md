# SwiftNet
The official PyTorch implementation of SwiftNet:Real-time Video Object Segmentation, which has been accepted by CVPR2021. 


## Requirements
 - Python >= 3.6
 - Pytorch 1.5
 - Numpy
 - Pillow
 - opencv-python
 - scipy
 - tqdm
 
## Training
 - The training pipeline of Swiftnet is similar with the training pipeline of [STM](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html), which can be found in our reproduced [STM training code](https://github.com/haochenheheda/Training-Code-of-STM).

## Inference
Usage
```
python eval.py -g 0 -y 17 -s val -D 'path to davis'
```

## Performance

Performance on Davis-17 val set.
| backbone | J&F | J |  F  | FPS | weights |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| resnet-18 | 77.6 | 75.5 | 79.7 | 65 | [`link`](https://drive.google.com/file/d/1I1agjrVIIUK6xU3pQF6wJ-TSvXEtg0kv/view?usp=sharing) |

Note:
	The FPS is tested on one P100, which does not include the time of image loading and evaluation cost.

## Acknowledgement
This repository is partially founded on the [official STM repository](https://github.com/seoungwugoh/STM).


## Citation
If you find this repository helpful and want to cite SwiftNet in your own projects, please use the following citation info.
```
@inproceedings{wang2021swiftnet,
  title={SwiftNet: Real-time Video Object Segmentation},
  author={Wang, Haochen and Jiang, Xiaolong and Ren, Haibing and Hu, Yao and Bai, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1296--1305},
  year={2021}
}
```
