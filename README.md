# PIS_score

This repository provides source code for the article *Association of artificial intelligence-based immunoscore with the efficacy of chemoimmunotherapy in patients with advanced nonsquamous non-small cell lung cancer (NSCLC): a multicentre retrospective study*.

# Installtation

The running environment of this repository base on  python3.11, CUDA >=11.8 and pytorch >=1.13.1

## Requirements

```bash
pip install -r requirements.txt
```

The code of `Feature_extraction` You need to download weights : [HistoSSLscaling](https://github.com/owkin/HistoSSLscaling?tab=readme-ov-file#feature-extraction)
Once the download is complete, execute the torch2jit.py code to get the jit weights


# CODE DESCRIPTION

## Feature_extraction
Please execute the code in order：
0_LocationBlockTable.py:The purpose is to get information about the position of the block

1_LocationBlockTable_visualization.py:Visualization of block information may be required to visualize a small number of samples to check that the image location information is correct

2_main_getFeature.py:Multi-process feature extraction

## MIL_method
fold123.xlsx: Trifold cross-validation training sample serial numbers and labels

t1_cfg.py:Configuration code files

t1_main.py:train code files

fold123_pred.py:Example code for fusion of weight prediction results from tri-fold cross validation training
# For academic research use only, please contact us at info@bio-totem for commercial use. 
