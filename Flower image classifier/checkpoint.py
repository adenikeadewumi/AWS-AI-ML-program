import os
import torch
import torch.nn as nn
def save_checkpoint(model: nn.Module,
                    model_name: str,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    epoch: int,
                    device_trained_on: torch.device,
                    hidden_layers: tuple,
                    output_layer: int,
                    classes: list,
                    class_to_idx: dict,
                    transforms,
                    checkpoint_name: str = "checkpoint") -> str:
    '''
    Function for saving the general checkpoint including anything important other than state_dict() of model.
​
    Args:
        optimizer: optimizer whose state_dict has to be saved
        criterion: loss/criterion used for training the network
        epoch: The epoch number of best performing model
        device_trained_on: The device on which the model was trained on
        hidden_layers: The specifications of the two hidden layers
        classes: The list of classes for the given classification problem
        class_to_idx: The class to index mapping/dictionary
        dheckpoint_name: The name by which to save the general checkpoint
​
    Returns:
        The path to saved general checkpoint (str)
    '''

    if not os.path.exists(checkpoint_name):
      os.makedirs(checkpoint_name)
    # checkpoint save location
    SAVE_PATH = os.path.join(checkpoint_name, "checkpoint.pth")

    # saving the checkpoint
    torch.save({"model_name": model_name,  # Store the model architecture name
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "criterion": criterion,
                "epoch": epoch,
                "device_trained_on": device_trained_on,
                "hidden_layers": hidden_layers,
                "output_layer": output_layer,
                "transforms": transforms,
                "classes": classes,
                "class_to_idx": class_to_idx}, SAVE_PATH)

    # printing save confirmation
    print(f"[INFO] The general checkpoint has been saved to: {SAVE_PATH}")

    # returning the saved checkpoint path
    return SAVE_PATH


# loading model
import torchvision.models as models
def load_model(checkpoint_save_path: str, available_device ) -> dict:
    '''
    Function for loading state_dict() of the saved model and inserting that into a newly defined architecture (must be same as original model arch).
​
    Args:
        model_name: Name of the trained model
        model_save_path: Save path to the state_dict() of model
        hidden_layers: The specifications of the two hidden layers
        output_layer: The specification of the output layer
        device_trained_on: The device on which the model was trained on originally
        device: The device on which the newly defined model should be moved to
​
    Returns:
        The model with loaded state_dict()
    '''

    if available_device:
      device= torch.device("cuda")
    else:
      device= torch.device("cpu")
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_save_path, map_location=device)
    model_name = checkpoint['model_name']

    # Define a dictionary mapping model names to their corresponding constructors
    model_dict = {
        "mobilenet": models.mobilenet_v3_large,
        "vit-b": models.vit_b_16,
        "efficientnet_b0": models.efficientnet_b0
    }


    # Ensure the model name exists in the dictionary
    if model_name not in model_dict:
        raise ValueError(f"Unknown model architecture: {model_name}")

    model = model_dict[model_name](weights= None)  # Set False since we load custom weights

    # Load the saved weights

    hidden_layer1= checkpoint["hidden_layers"][0]
    hidden_layer2= checkpoint["hidden_layers"][1]
    output_layer= checkpoint["output_layer"]
    device_trained_on= checkpoint["device_trained_on"]
    
    in_features = model.classifier[0].in_features

    model.classifier= nn.Sequential(nn.Linear(in_features, hidden_layer1),
                        nn.ReLU(),
                        nn.Linear(hidden_layer1, hidden_layer2),
                        nn.ReLU(),
                        nn.Linear(hidden_layer2, output_layer)
                       )
    model.load_state_dict(checkpoint["model_state_dict"])

    """# loading model state_dict
    if str(device_trained_on) in ("cuda", "cuda:0"):
        # saved on GPU, loading on CPU case
        if str(device) == "cpu":
            model.load_state_dict(torch.load(checkpoint["model_state_dict"], map_location=device))
        # saved on GPU, loading on GPU case
        elif str(device) in ("cuda", "cuda:0"):
            model.load_state_dict(torch.load(checkpoint["model_state_dict"]))
    else:
        # saved on CPU, loading on CPU case
        if str(device) == "cpu":
            model.load_state_dict(torch.load(checkpoint["model_state_dict"]))
        # saved on CPU, loading on GPU case
        elif str(device) in ("cuda", "cuda:0"):
            model.load_state_dict(torch.load(checkpoint["model_state_dict"], map_location="cuda:0"))"""

    # moving model to device and switching to eval 
    for param in model.parameters():
      param.data= param.data.to(torch.float)
    model.to(device);
    model.eval();


    transform= checkpoint["transforms"]
    classes= checkpoint["classes"]
    class_to_idx= checkpoint["class_to_idx"]

    # returning model
    return model, transform, classes, class_to_idx
