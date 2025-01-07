# Identity-Preserving Face Swapping via Dual Surrogate Generative Models

<a href='https://bone-11.github.io/cs-cs/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; [![Paper Link](https://img.shields.io/badge/Paper-Doi-red)](https://dl.acm.org/doi/10.1145/3676165)

This is the repository of the paper ***Identity-Preserving Face Swapping via Dual Surrogate Generative Models***. For now, we upload the inference code and checkpoint.

## Getting Started

### Environment

```shell
pip install -r requirements.txt
```

Then download ID encoder weight ms1mv3_arcface_r100_fp16_backbone.pth from our upload
[one drive](https://1drv.ms/f/c/64d71f39113d98e4/Eg6nvnA849VAjxFTdh6opXkBXcxB7LQg2w1iwHV2QXyY2Q?e=1BfqpE)
and should be placed in *./model/arcface/*

### Inference Checkpoints

You can download the checkpoints from [one drive](https://1drv.ms/f/c/64d71f39113d98e4/ElBkLV2YQXdHgJbsc2Aboy8BBhhvct14hvW8sGD87F2Nzg?e=U2Yqxj) and place them at *./*.

## Inference

Before swapping, use facealign.sh to align the face images.

After alignment, inference_adapter.sh is utilized to swapping

```shell
bash facealign.sh
bash inference_adapter.sh
```

## Training

Download the training data from [one drive](https://1drv.ms/f/c/64d71f39113d98e4/El8ChUj0d5BIk5yMGkiyR8kB450SvhZYY6d4sm5sksZIeA?e=p4Dk8T) and place them at ./train_data. Then run the following scirpt

```shell
bash train_adapter.sh
```

And the results can be found in ./expr/train_smswap_faceshifter_adapter.

## License and Citation
CSCS is released only for academic research. Researchers are allowed to use this code and weights freely for non-commercial purposes.

**Reference Format:**
```
@article{huang2024cscs,
  title={Identity-Preserving Face Swapping via Dual Surrogate Generative Models},
  author={Huang, Ziyao and Tang, Fan and Zhang, Yong and Cao, Juan and Li, Chengyu and Tang, Sheng and Li, Jintao and Lee, Tong-Yee},
  journal={ACM Transactions on Graphics},
  volume={43},
  number={5},
  pages={1--19},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

