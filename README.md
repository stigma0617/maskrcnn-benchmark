## Impelmentation of Detectron with VoVNet Backbone Networks

This repository contains [Detectron](https://github.com/facebookresearch/maskrcnn-benchmark) with [VoVNet](https://arxiv.org/abs/1904.09730) (CVPRW'19) Backbone Networks. This code based on pytorch imeplementation of Detectron ([maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)) 

### Hilights

- Memory efficient 
- Better performance, especially for small object
- Faster speed



### Comparison with ResNet backbones

- 2x schedule
- same hyperparameters
- same training protocols ( max epoch, learning rate schedule, etc)
- NOT multi-scale training augmentation
- 8 x TITAN Xp GPU
- pytorch1.1
- CUDA v9
- cuDNN v7.2


| Backbone | Detector | Train mem(GB) | Inference time (ms) | Box AP (AP/APs/APm/APl) | Mask AP (AP/APs/APm/APl) | DOWNLOAD |
|----------|----------|---------------|:-------------------:|:------------------------:|:--------------------------:| :---:|
| R-50     | Faster   | 3.6           | 78                  | 37.5/21.3/40.3/49.5      | -                          |[link](https://www.dropbox.com/s/kmcfd0j3cn9gevz/FRCN-R-50-FPN-2x.pth)|
 R-101    | Faster   | 4.7           | 97                  | 39.6/22.8/43.2/51.9      | -                          |[link](https://www.dropbox.com/s/wzohk5zm9e7xw7k/FRCN-R-101-FPN-2x.pth)|
| **V-39**     | Faster   | 3.9           | 78                  | 39.8/23.7/42.6/51.5      | -                          |[link](https://www.dropbox.com/s/svg9ynha9l9oqp0/FRCN-V-39-FPN-2x.pth)|
| **V-57**     | Faster   | 4.4           | 87                  | 40.8/24.8/43.8/52.4      | -                          |[link](https://www.dropbox.com/s/fawts3l0idznvvb/FRCN-V-57-FPN-2x.pth)|
| R-50     | Mask     | 3.6           | 83                  | 38.6/22.1/41.3/51.4      | 34.9/16.0/37.3/52.2        |[link](https://www.dropbox.com/s/dmkcu8dc662nnsu/MRCN-R-50-FPN-2x.pth)|
| R-101    | Mask     | 4.7           | 102                 | 40.8/23.2/44.0/53.9      | 36.7/16.7/39.4/54.3        |[link](https://www.dropbox.com/s/0k73qa5b8fpb45h/MRCN-R-101-FPN-2x.pth)|
| **V-39**     | Mask     | 4             | 81                  | 41.0/24.6/43.9/53.1      | 36.7/17.9/39.3/53.0        |[link](https://www.dropbox.com/s/3zpmq4nvijqek3m/MRCN-V-39-FPN-2x.pth)|
| **V-57**     | Mask     | 4.5           | 90                  | 42.0/25.1/44.9/53.8      | 37.5/18.3/39.8/54.3        |[link](https://www.dropbox.com/s/3zpmq4nvijqek3m/MRCN-V-39-FPN-2x.pth)|



### ImageNet pretrained weight

-[VoVNet-39](https://www.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth)
-[VoVNet-57](https://www.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth)


### Preparation


```bash
git clone https://github.com/stigma0617/maskrcnn-benchmark-vovnet.git
cd maskrcnn_benchmark-vovnet
checkout vovnet

mkdir -p checkpoints/pretrained
cd checkpoints/pretrained
wget https://www.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth
wget https://www.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth
```


### Installation

Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)





### Training & Inferecne

Follow [the instructions](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training) [maskrcnn-benchmark](https://github.com/facebookresearch) guides

For example,

```bash
cd maskrcnn_benchmark-vovnet
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/vovnet/e2e_faster_rcnn_V_39_FPN_2x.yaml" 
```