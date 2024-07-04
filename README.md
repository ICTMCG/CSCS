# Identity-Preserving Face Swapping via Dual Surrogate Generative Models

<a href='https://bone-11.github.io/cs-cs/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; [![Paper Link](https://img.shields.io/badge/Paper-Doi-red)](https://dl.acm.org/doi/10.1145/3676165)

This is the repository of the paper ***Identity-Preserving Face Swapping via Dual Surrogate Generative Models***. For now, we upload the inference code and checkpoint.

## Getting Started

### Environment

```shell
pip install -r requirements.txt
```

Then download ID encoder weight ms1mv3_arcface_r100_fp16_backbone.pth from:

https://onedrive.live.com/?id=4A83B6B633B029CC!5577&resid=4A83B6B633B029CC!5577&authkey=!AFZjr283nwZHqbA&cid=4a83b6b633b029cc

and should be placed in *./model/arcface/*

### Inference Checkpoints

You can download the checkpoints from [https://1drv.ms/f/c/64d71f39113d98e4/ElBkLV2YQXdHgJbsc2Aboy8B979bVu6ilvcYxbiGOWClLQ?e=n647fU] and place them at *./*.

## Inference

Before swapping, use facealign.sh to align the face images.

After alignment, inference_adapter.sh is utilized to swapping

```shell
bash facealign.sh
bash inference_adapter.sh
```

