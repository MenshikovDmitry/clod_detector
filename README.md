# Clod Detector
**!!! This repository has not been tested in Windows environment !!!**


![demo gif](demo.gif?raw=true "Pic")
[Original video](https://drive.google.com/file/d/1yEiiakDfC5v6omWZgEPXK8ji182_KfGC/view?usp=sharing)

### Installation:
Clone, create virtual environment and install dependencies:
```sh
$ git clone https://github.com/MenshikovDmitry/clod_detector
$ cd clod_detector
$ virtualenv -p python3.6 .env
$ source .env/bin/activate
(.env) $ pip install -r requirements.txt
```

**All further actions are performed within the virtual environment**

Download weights:
```sh
$wget https://s3-ap-southeast-2.amazonaws.com/menshikov.info/mask_rcnn_komok.h5
```

### Demo
```bash
(.env)$ python komok_detector.py --weights=mask_rcnn_komok.h5 <image or video file>
```
press ESC to exit the demo


### Creating dataset from video file
Creating a number of screenshots from the video file.
Check the settings inside the _create_dataset.py_ file, it contains hardcoded parameters.
```sh
(.env)$ python create_dataset.py
```

### Training the model with generated dataset. 
Assumes that there is a file with labels (check the repo).
Starts with a model pretrained on COCO dataset.
```sh
(.env)$ custom.py train --dataset=/path/to/datasetfolder --weights=coco
```
It will create weights in **log** folder.


