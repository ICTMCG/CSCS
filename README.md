# Identity-Preserving Face Swapping via Dual Surrogate Generative Models

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

We have uploaded the inference checkpoint in this repository via git-lfs, and it can be downloaded by:

```shell
git lfs install
git clone https://github.com/bone-11/cscs.git
```

## Inference

Before swapping, use facealign.sh to align the face images.

After alignment, inference_adapter.sh is utilized to swapping

```shell
bash facealign.sh
bash inference_adapter.sh
```

