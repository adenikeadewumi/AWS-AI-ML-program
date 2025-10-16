# Import necessary modules and functions
import argparse
from data_utils import build_model, dataloader
from model_utils import training, manual_seed  # Example imports
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from checkpoint import save_checkpoint
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="mobilenet", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hiddenlayer1", type=int, default=512, help="Number of neurons/units in first hidden layer")
    parser.add_argument("--hiddenlayer2", type=int, default=256, help="Number of neurons/units in second hidden layer")
    parser.add_argument("--output_layer", type=int, default=102, help="Number of classes to be predicted")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()


    model,train_transforms, val_test_transforms= build_model(args.arch, (args.hiddenlayer1, args.hiddenlayer2), args.output_layer)
    trainloader, validloader, testloader = dataloader(args.data_directory, train_transforms, val_test_transforms)

    CLASSES= trainloader.dataset.classes
    CLASS_TO_IDX= trainloader.dataset.class_to_idx
    NUM_CLASSES= len(CLASSES)

    train_accuracy= MulticlassAccuracy(num_classes= NUM_CLASSES)
    train_f1= MulticlassF1Score(num_classes=  NUM_CLASSES)
    val_accuracy= MulticlassAccuracy(num_classes= NUM_CLASSES)
    val_f1= MulticlassF1Score(num_classes= NUM_CLASSES)
    manual_seed()
    model_state_dict, training_stat, device= training(model, trainloader,validloader,torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate),nn.CrossEntropyLoss(), train_accuracy, train_f1, val_accuracy, val_f1, args.gpu, args.epochs)

    # Save the checkpoint
    checkpoint_save_path= save_checkpoint(model, args.arch, torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate), nn.CrossEntropyLoss(), args.epochs, torch.device, (args.hiddenlayer1, args.hiddenlayer2), args.output_layer, CLASSES, CLASS_TO_IDX, val_test_transforms, args.save_dir)

if __name__ == '__main__':
    main()
