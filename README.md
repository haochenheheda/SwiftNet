# SwiftNet
Implementation of SwiftNet:Real-time Video Object Segmentation.


## Requirements
 - Python >= 3.6
 - Pytorch 1.5
 - Numpy
 - Pillow
 - opencv-python
 - scipy
 - tqdm
 
## Training
 - The training pipeline of Swiftnet is similar with the training pipeline of [STM](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html). You could refer to our reproduced [STM training code](https://github.com/haochenheheda/Training-Code-of-STM).

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
	The fps is tested on one P100, which does not include the time of image loading and evaluation.

## Acknowledgement
This codebase borrows the code and structure from [official STM repository](https://github.com/seoungwugoh/STM).


## Citation

```
@article{wang2021swiftnet,
  title={SwiftNet: Real-time Video Object Segmentation},
  author={Wang, Haochen and Jiang, Xiaolong and Ren, Haibing and Hu, Yao and Bai, Song},
  journal={arXiv preprint arXiv:2102.04604},
  year={2021}
}
```
