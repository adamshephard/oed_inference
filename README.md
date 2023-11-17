# Transformer-based Model for Oral Epithelial Dysplasia Segmentation

This repository provides the inference code for the models used for segmenting dysplasia in OED. Link to preprint [here](https://arxiv.org/abs/2311.05452). <br />

We have used the TIAToolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) Semantic Segmentor Engine in the below script for processing. 

## Set Up Environment

We use Python 3.10 with the [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) package installed. By default this uses PyTorch 2.0.

```
conda create -n tiatoolbox python=3.10
conda activate tiatoolbox
pip install tiatoolbox
pip install ml_collections
pip install termcolor
```

## Repository Structure

Below are the main directories in the repository: 

- `models/`: model definition

Below are the main executable scripts in the repository:

- `run_segmentation.py`: transunet inference script

## Inference

### Data Format
Input: <br />
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

### Model Weights

Model weights obtained from training on the Sheffield OED dataset: [OED checkpoint](https://drive.google.com/file/d/1EF3ItKmYhtdOy5aV9CJZ0a-g03LDaVy4/view?usp=sharing
)
If any of the checkpoints are used, please ensure to cite the corresponding paper.

### Usage

#### Segmentation with TransUNet via TIAToolbox

To run TransUNet on images/WSIs to generate dysplasia segmentation maps we do the following. This should be quite fast, when run on a GPU as is done at 1.0mpp. Ensure to change the `input_dir`, `output_dir`, and `weights_path` arguments within the file to ensure they are pointing towards the correct directories/files.

Usage: <br />
```
  python run_segmentation.py
```