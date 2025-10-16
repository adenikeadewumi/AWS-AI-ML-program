import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import os
def build_model(arch: str, hiddenlayers, output_size):
  available_models= {"mobilenet": (torchvision.models.MobileNet_V3_Large_Weights,
                                   torchvision.models.mobilenet_v3_large),
                     "efficientnet _v2": (torchvision.models.EfficientNet_V2_S_Weights,
                                          torchvision.models.efficientnet_v2_s),
                     "vit-b": (torchvision.models.ViT_B_16_Weights,
                               torchvision.models.vit_b_16)}
  if arch.lower() not in available_models.keys():
    print(f"{arch} is not a supported model. Please choose from {available_models.keys()}")

  if arch.lower() == "mobilenet":
    mobilenet_weights= available_models[arch.lower()][0].DEFAULT
    transform= mobilenet_weights.transforms()
    model= available_models[arch.lower()][1](weights= mobilenet_weights)
  elif arch.lower() == "efficientnet _v2":
    efficientnet_weights= available_models[arch.lower()][0].DEFAULT
    transform= efficientnet_weights.transforms()
    model= available_models[arch.lower()][1](weights= efficientnet_weights)
  elif arch.lower() == "vit-b":
    vit_weights= available_models[arch.lower()][0].DEFAULT
    transform= vit_weights.transforms()
    model= available_models[arch.lower()][1](weights= vit_weights)

  for params in model.parameters():
    params.require_grad= False
  if hasattr(model, "classifier"):
    input_features = model.classifier[0].in_features  # Works for models like MobileNet
  elif hasattr(model, "fc"):
    input_features = model.fc.in_features  # Works for models like ResNet, ViT, etc.
  else:
    raise ValueError("Cannot determine input feature size for this model.")
  model.classifier= nn.Sequential(nn.Linear(input_features, hiddenlayers[0]),
                                  nn.ReLU(),
                                  nn.Linear(hiddenlayers[0], hiddenlayers[1]),
                                  nn.ReLU(),
                                  nn.Linear(hiddenlayers[1], output_size))
  crop_size= transform.crop_size[0]
  resize_value= transform.resize_size[0]
  mean= transform.mean[0]
  std= transform.std[0]

  train_transforms = transforms.Compose([
    transforms.Resize((resize_value, resize_value), interpolation=transforms.InterpolationMode.BILINEAR),  # Resize slightly larger
    transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),  # Random augmentation
    transforms.CenterCrop((crop_size, crop_size)),  # Crop to crop_size
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
     ])
  val_test_transforms = transforms.Compose([
    transforms.Resize((resize_value, resize_value), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((crop_size, crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
  return model, train_transforms, val_test_transforms


def dataloader(data_dir,train_transforms, val_test_transforms):
  train_path= os.path.join(data_dir, "train")
  valid_path= os.path.join(data_dir, "valid")
  test_path= os.path.join(data_dir, "test")

  # Create the image dataset for training, validation, testing
  train_dataset = torchvision.datasets.ImageFolder(root= str(train_path), transform= train_transforms)
  valid_dataset = torchvision.datasets.ImageFolder(root= str(valid_path), transform= val_test_transforms)
  test_dataset = torchvision.datasets.ImageFolder(root= str(test_path), transform= val_test_transforms)
  #definr the dataloaders
  train_dataloaders = DataLoader(train_dataset, batch_size= 32, shuffle= True)
  valid_dataloaders = DataLoader(valid_dataset, batch_size= 32, shuffle= False)
  test_dataloaders = DataLoader(test_dataset, batch_size= 32, shuffle= False)
  return train_dataloaders, valid_dataloaders, test_dataloaders



