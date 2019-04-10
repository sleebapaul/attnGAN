**First things first**  
This repo is a convenient clone of AttnGAN by MSFT Research. You can find the original repo at [here](https://github.com/taoxugit/AttnGAN.git). 

AttnGAN original repo contains the training and testing code. But this repo is solely intended to use for generating images from text using MSCOCO pretrained weights. 

# AttnGAN

Pytorch implementation for reproducing AttnGAN results in the paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). 

<img src="framework.png" width="900px" height="350px"/>


**Python version**

`python 2.7`

**Dependencies**

- `pytorch`
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`

**YML files**
- `*.yml` files are example configuration files for training/evaluation our models.


**Pretrained Model**
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/coco/`
- [AttnGAN for coco](https://drive.google.com/open?id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi). Download and save it to `models/`

**Setup script**

Run the following script for setting up the dependencies.

```
./setup.sh
```

**Sampling**
- Run the command from project folder.
```
python2 gen_art.py --gpu 0 --input_text "Mary had a little lamb" --data_dir data/coco --model_path models/coco_AttnGAN2.pth --textencoder_path DAMSMencoders/coco/text_encoder100.pth --output_dir output` 
```
- Output will be stored into `output_dir` folder. 
- Change the `eval_*.yml` files to generate images from other pre-trained models. Default config file is `cfg/eval_coco.yml`. 

