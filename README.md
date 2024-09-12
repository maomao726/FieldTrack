# FieldTrack

# Requirements
* Python, Pytorch>2.0.0, torchvision>0.15.0

(We are using python==3.10.14, torch==2.2.0, torchvision==0.17.0, FYI)

```bash
# installing tiny package of football_field
$ cd football_field
$ python setup.py develop

# install motified motmetrics
$ cd ../motmetrics
$ python setup.py develop

# install required packages and yolox
$ cd FieldTrack
$ pip install -r requirements.txt
$ pip install lapx
$ python setup.py develop

# install torchreid
$ cd external/reid
$ python setup.py develop

#install fast-reid
$ cd ../fast-reid
$ python setup.py develop
```

# Pretrained Weight
* YOLOX

    You can download pretrained yolox weight from [OC-SORT](https://github.com/noahcao/OC_SORT.git) and put them in `FieldTrack/pretrained`.

* torchreid

    You can download pretrained reid weight from [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT.git) and put them in `FieldTrack/external/weights/`.

* (optional) Keypoints Detector

    The file size is too large, so DM me if you need a pretrained weights on football data, or train your own weights with my another repository [football_extension](https://github.com/maomao726/football_extension.git)

# Usage
```bash
$ cd FieldTrack
$ python eval_football_track.py
```
* It works with loading config file `FieldTrack/config/eval_my.py`. You can modified any parameter inside if you need.

# evaluation
