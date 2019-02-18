# Sidewalk-semantic-seg
Semantic segmentation for sidewalks. It has different models and options to train over. 

## Train From Scratch
Run this in the parent directory

```
python train.py --model=Model_name --num_epochs=20 --crop_height=128 --crop_width=192 --checkpoint_step=1 --dataset=CityScape
```
Training parameters
``` 
num_epochs= Number of epochs to train
crop_height= 128
crop_width= 192
checkpoint_step= Number of epochs to perform before saving latest checkpoint
dataset= Name of directory with data. Data directory should be in parent directory
batch_size = Batch size
model = FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, MobileUNet, MobileUNet-Skip, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet, FRRN-A, FRRN-B, PSPNet, GCN, DeepLabV3, DeepLabV3_plus, AdapNet, DenseASPP, DDSC,  BiSeNet, custom
```
Note: For model option custom, you have to code your own model. 

## Train from pretrained weights 

Run this in parent directory

```
python train.py --frontend=Model_name --num_epochs=10 --crop_height=128 --crop_width=192 --checkpoint_step=1 --dataset=CityScape
```

All parameters are the same except for frontend

```
frontend = ResNet50, ResNet101, ResNet152, MobileNetV2, InceptionV4, SEResNeXt50, SEResNeXt101
```

## Link to labeled dataset

Link: 

**Note**: Put the dataset in the parent directory. It should look like this 

Parent Directory
  |
  |
  Cityscape
      |
      test
          |
          Image 1
          Image 2
      |
      test_labels
          |  
          Image 1
          Image2
      |
      |
      train
        |
        Image1 Image2
      |
      |
      train_labels
      |
      |
      val
      |
      |
      val_labels
      |
      |
      classes.csv
      
      

### Dataset attributes

Sidewalk color: rgb(0,0,192)
All other color: rgb(0,0,0)








