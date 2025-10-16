import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json


def predict_and_plot_topk(model: nn.Module,
                          img_transform: transforms.Compose,
                          class_list: list,
                          image_path: str,
                          available_device,
                          category_names,
                          class_to_idx,
                          topk: int = 5):

    if available_device:
      device= torch.device("cuda")
    else:
      device= torch.device("cpu")
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = img_transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Moving model to device and switching to eval mode

    model.to(device)
    model.eval()

    # Make predictions
    with torch.inference_mode():
        output = model(input_batch)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top-k class indices and probabilities
    topk_probs, topk_indices = torch.topk(probabilities, topk)
    topk_probs_np = topk_probs.numpy(force=True)
    
    with open(category_names, "r") as f:
        cat_to_name = json.load(f)
    labels= []
    probs= []
    for i in range(len(topk_indices)):
        pred_indices= topk_indices[i].item() #int(topk_indices[i].numpy(force=True))
        key = [key for key, val in class_to_idx.items() if val == pred_indices]
        labels.append(cat_to_name.get(str(key[0]), key[0]))
        probs.append(topk_probs[i])

    final_result= list(zip(labels, probs))

    topk_indices_np = topk_indices.numpy(force=True)
    #print("Prnting topk indices")
    #print(topk_indices)
    #print(topk_probs)

    # Convert tensor to numpy array for plotting
    probs_np = probabilities.numpy(force=True)
    #print("Print probs_np")
    #print(probs_np)

    # Create a horizontal bar graph
    plt.figure(figsize=(10, 6))

    # Plot the image
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Plot the top-k classes
    plt.subplot(2, 1, 2)
    plt.barh([class_list[i] for i in topk_indices_np], topk_probs_np, color='blue')
    plt.xlabel('Predicted Probability')
    plt.title(f'Top-{topk} Predicted Classes')

    plt.tight_layout()
    plt.show()

    return final_result

