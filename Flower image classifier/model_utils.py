#from torchinfo import summary
import torch.nn as nn
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm.autonotebook import tqdm
import copy
from typing import Tuple, Callable, Union, Any

def training_step(model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                training_accuracy: MulticlassAccuracy,
                training_f1: MulticlassF1Score,
                epoch_num: int,
                available_device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #CHeck which device is available
    if available_device:
        device= torch.device("cuda")
    else:
        device= torch.device("cpu")
    # Set model to train mode
    model.train()
    # Resetting torchmetrics values
    training_accuracy.reset()
    training_f1.reset()
    epoch_loss= 0
    with tqdm(enumerate(train_dataloader), total= len(train_dataloader), unit= "train-batch") as tepoch:
        for batch_idx, (features, labels) in tepoch:
          tepoch.set_description(f"Epoch: {epoch_num+1}  Phase - Training")
          features, labels = features.to(device), labels.to(device).type(torch.long)
          model.to(device)
          logits= model(features)
          loss= criterion(logits, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          with torch.no_grad():
            avg_loss_sample= loss/len(labels)
            batch_acc= training_accuracy.forward(logits, labels)
            batch_f1= training_f1.forward(logits, labels)

          # accumulating epoch loss
          epoch_loss+= avg_loss_sample

          # Setting postfix for progress bar
          tepoch.set_postfix(batch_f1_score= batch_f1.item(),  batch_accuracy=f"{batch_acc.item()*100:.2f}%", loss_per_sample=avg_loss_sample.item())
        epoch_loss/= len(train_dataloader)
    print(f"[INFO] Epoch: {epoch_num+1}  |  loss: {epoch_loss.item():.3f} | training acc: {training_accuracy.compute().item()*100:.2f}%   | training f1-score: {training_f1.compute().item():.2f}")

    return epoch_loss, training_accuracy.compute(), training_f1.compute()




def testing_step(model: nn.Module,
                 test_dataloader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 test_accuracy: MulticlassAccuracy,
                 test_f1: MulticlassF1Score,
                 epoch_num: int,
                 available_device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Check for available device
    if available_device:
        device= torch.device("cuda")
    else:
        device= torch.device("cpu")
    # settting model tp train mode
    model.eval()


    # varible for calculting the epoch loss
    epoch_loss= 0

    #Use tqdm for progress bar
    with tqdm(enumerate(test_dataloader, start= 1), total= len(test_dataloader), unit= "test-batch") as tepoch:
        # iterating over the data batches
        for batch_idx, (features, labels) in tepoch:
            # setting description for tqdm progress bar
            tepoch.set_description(f'Epoch: {epoch_num+1} | Phase - Validation/Test')

            # moving the batches to device
            features, labels = features.to(device), labels.to(device).type(torch.long)

            with torch.no_grad():
                logits= model(features)

                loss= criterion(logits, labels)
                avg_loss_sample= loss/len(labels)
                batch_acc= test_accuracy.forward(logits, labels)
                batch_f1= test_f1.forward(logits, labels)
            # accumulating in epoch loss
            epoch_loss+= avg_loss_sample

            # setting postfix for progress bar
            tepoch.set_postfix(batch_f1_score= batch_f1.item(),  batch_accuracy=f"{batch_acc.item()*100:.2f}%", loss_per_sample=avg_loss_sample.item())

    epoch_loss/= len(test_dataloader)

    print()

     # printing epoch statistics
    print(f"[INFO] Epoch: {epoch_num+1} | loss: {epoch_loss.item():.3f} | val/test acc: {test_accuracy.compute().item()*100:.2f}% | val/test f1-score: {test_f1.compute().item():.2f}")


    # returning loss, acc and f1
    return epoch_loss, test_accuracy.compute(), test_f1.compute()

def training(model: nn.Module,
             train_dataloader: torch.utils.data.DataLoader,
             test_dataloader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             criterion: nn.Module,
             training_accuracy: MulticlassAccuracy,
             training_f1: MulticlassF1Score,
             test_accuracy: MulticlassAccuracy,
             test_f1: MulticlassF1Score,
             available_device,
             epochs: int= 5)-> Tuple[dict, dict]:
  # Check for available device
  if available_device:
    device= torch.device("cuda")
  else:
    device= torch.device("cpu")
  # Convert paramters to device
  model.to(device)
  training_accuracy.to(device)
  training_f1.to(device)
  test_accuracy.to(device)
  test_f1.to(device)
  # Create empty lists
  train_loss = []
  train_acc = []
  train_f1 = []
  val_loss = []
  val_acc = []
  val_f1 = []

  best_loss= float("inf")

  for epoch_interation in range(epochs):
    train_epoch_loss, train_epoch_acc, train_epoch_f1= training_step(model,
                                                                     train_dataloader,
                                                                     optimizer,
                                                                     criterion,
                                                                     training_accuracy,
                                                                     training_f1,
                                                                     epoch_interation,
                                                                     device)
    val_epoch_loss, val_epoch_acc, val_epoch_f1= testing_step(model,
                                                                 test_dataloader,
                                                                 criterion,
                                                                 test_accuracy,
                                                                 test_f1,
                                                                 epoch_interation,
                                                                 device)
    #Append the training statitics to the list predefined
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)
    train_f1.append(train_epoch_f1)

    #Append the testing statitics to the list predefined
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)
    val_f1.append(val_epoch_f1)

    print()

    if val_epoch_loss < best_loss:
         best_loss= val_epoch_loss
         best_model= copy.deepcopy(model)
         best_epoch= epoch_interation

  train_loss = [tensor.item() for tensor in train_loss]
  train_acc = [tensor.item() for tensor in train_acc]
  train_f1 = [tensor.item() for tensor in train_f1]
  val_loss = [tensor.item() for tensor in val_loss]
  val_acc = [tensor.item() for tensor in val_acc]
  val_f1 = [tensor.item() for tensor in val_f1]








  training_stats= {"training_loss": train_loss,
                     "training_accuracy": train_acc,
                     "training_f1": train_f1,
                      "validation_loss": val_loss,
                      "validation_accuracy": val_acc,
                      "validation_f1": val_f1,
                      "best_epoch": best_epoch+1,
                      "loss_on_best_epoch": best_loss.item(),
                      "device_used": device}


  return best_model, training_stats, device

def manual_seed(random_seed: int= 42)-> None:
    """For maintaining reproducibilit of a notebook cell"""
    #for non_cuda device
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

#model, train_transform, val_test_transform= build_model("mobilenet", (512, 128), 102)
#trainloader, validloader, testloader= dataloader(FLOWERS_DIR, train_transform, val_test_transform)

#CLASSES= trainloader.dataset.classes
#CLASS_TO_IDX= trainloader.dataset.class_to_idx
#NUM_CLASSES= len(CLASSES)

#train_accuracy= MulticlassAccuracy(num_classes= NUM_CLASSES)
#train_f1= MulticlassF1Score(num_classes=  NUM_CLASSES)
#val_accuracy= MulticlassAccuracy(num_classes= NUM_CLASSES)
#val_f1= MulticlassF1Score(num_classes= NUM_CLASSES)
#manual_seed()

#LEARNING_RATE= 0.001
#model_state_dict, training_stat= training(model, trainloader,validloader,torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE),nn.CrossEntropyLoss(), train_accuracy, train_f1, val_accuracy, val_f1)
