# Import necessary modules and functions
import argparse
import json
import torch
from checkpoint import load_model
from pred_utils import predict_and_plot_topk

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()
    print(args.checkpoint)
    # Load the model from checkpoint
    model, transform, CLASSES, class_to_idx  = load_model(args.checkpoint, args.gpu)
    print(type(model))
    # Process and Predict the class

    final_result = predict_and_plot_topk(model, transform, CLASSES, args.input, args.gpu, args.category_names, class_to_idx, args.top_k)
    

    print()
    print("Predictictions: ")
    for flower, prob in final_result:
      print(f"{flower} : {prob}")

if __name__ == '__main__':
    main()
