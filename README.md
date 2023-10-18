# the-detection-and-classifing-of-NF1


## Introduction


This project is based on yolov5 and transformer's self-attention mechanism as architecture developed project to detect and classify neuroma images. The self-attention mechanism is embedded into the backbone of yolov5 as a way to improve accuracy and inference speed.

### Environment Configuration


The detailed environment configuration is in requirements.txt as follows.
```shell
pip install -r requirements.txt
```


### Data set processing

For data processing, we first label the original image, we choose to use labelme to label the image. We can get the labeled json file, and later we will use ```.dataset/NF1_data/n2coco.py``` to convert the labelme json file to json file in coco dataset format. Immediately after that, we will use ```.dataset/NF1_data/NF1_data.py``` to convert the json file to text format to conform to yolov5's annotation input format. And then we will divide the dataset into training set and validation set. (This will require the use of ```./dataset/NF1_data/distribution_dataset.py```)

### Training module

#### Self-attention mechanism


### Inference module

### Analysis of results
