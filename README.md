# Thesis Project - Tiny-PointNet the Semantic Segmentation framework for point clouds

## Tiny-PointNet model
![model][model]
```
Total params: 238,813
Trainable params: 236,765
Non-trainable params: 2,048

Model size: ~5 MB
Model weights size: ~3 MB
```


> ### Pretrained models and other assets
> The ```assets/pretrained_models``` folder contains both trained models and their weights.
>
> The ```assets/TinyPointNet-<DATASET_NAME>``` folders contains (in compressed form) the training logs, the model weights of each epoch, Google Colab notebook and the saved model wich same as the ```assets/pretrained_models/tiny-pointnet-<DATASET_NAME>``` folders content.

## Using Tiny-PointNet framework
### Clone GitHub repository
```
$> git clone https://github.com/david-horvath/thesis-tiny-pointnet.git
```

### Install required libraries
```
$> cd thesis-tiny-pointnet
$> pip install -r requirements.txt
```

### **Download datasets ( Important! )**
> **First, please download the datasets using the links below and place them (zip files) in the ```datas``` folder.**
> * Download [VKITTI3D Dataset](https://drive.google.com/file/d/1QFMaKL5znKwCQmpmlL8o8tHatleYfC-H/view?usp=sharing)
> * Download [S3DIS Dataset](https://drive.google.com/file/d/1Mxqv-LJ976_R7YFzabws-ZOQ0yWQgctJ/view?usp=sharing)


### Train model on VKITTI3D dataset (accuracy: ~96%)
```
$> python train.py --num_points 4096 \
                   --num_classes 13 \
                   --batch_size 64 \
                   --epochs 100 \
                   --lr 0.001 \
                   --dataset vkitti3d \
                   --augment \
                   --eval
```
#### Training accuracy and loss
![pointnet-training-VKITTI3D][pointnet-training-VKITTI3D]

### Train model on S3DIS dataset (accuracy: ~84%)
```
$> python train.py --num_points 4096 \
                   --num_classes 13 \
                   --batch_size 64 \
                   --epochs 100 \
                   --lr 0.001 \
                   --dataset s3dis \
                   --augment \
                   --eval
```
#### Training accuracy and loss
![pointnet-training-S3DIS][pointnet-training-S3DIS]


### Predict point clouds
#### Predict point cloud from VKITTI3D dataset
```
$> python prediction.py --num_points 4096 \
                        --num_classes 13 \
                        --weights ./assets/pretrained_models/tiny-pointnet-vkitti3d/saved_weights.hdf5 \
                        --file ./datas/vkitti3d_dataset/vkitti3d_h5_v10001_00000.h5 \
                        --output ./predictions
```
#### Predict point cloud from S3DIS dataset
```
$> python prediction.py --num_points 4096 \
                        --num_classes 13 \
                        --weights ./assets/pretrained_models/tiny-pointnet-s3dis/saved_weights.hdf5 \
                        --file ./datas/s3dis_dataset/Area_1_conferenceRoom_1.h5 \
                        --output ./predictions
```


### Visualize predictions
#### Visualize predicted point cloud (VKITTI3D)
```
$> python visualization.py --file ./predictions/prediction_vkitti3d_h5_v10001_00000.h5 \
                           --semantic_classes vkitti3d
```
![prediction-vkitti3d][prediction-vkitti3d]
#### Visualize predicted point cloud (S3DIS)
```
$> python visualization.py --file ./predictions/prediction_Area_1_conferenceRoom_1.h5 \
                           --semantic_classes s3dis
```
![prediction-s3dis][prediction-s3dis]


[model]: assets/readme/model.png
[pointnet-training-VKITTI3D]: assets/readme/pointnet-training-VKITTI3D.png
[pointnet-training-S3DIS]: assets/readme/pointnet-training-S3DIS.png
[prediction-vkitti3d]: assets/readme/visu_vkitti3d.png
[prediction-s3dis]: assets/readme/visu_s3dis.png