import argparse
import os

import torch
from transformers import AutoModelForQuestionAnswering


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_of_pt_model",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model"
    )
    # Required parameters
    parser.add_argument(
        "--path_of_input_data",
        default=None,
        type=str,
        required=True,
        help="The path of the input data ([input_ids_data.txt, attention_mask_data.txt, token_type_ids_data.txt]",
    )
    # Required parameters
    parser.add_argument(
        "--path_of_output",
        default=None,
        type=str,
        required=True,
        help="The path of output folder. The model would be exported as 'path_of_output/model.onnx'",
    )

    args = parser.parse_args()

    model = AutoModelForQuestionAnswering.from_pretrained(args.path_of_pt_model)
    path_of_input_data = args.path_of_input_data

    import pickle
    f1 = open(os.path.join(path_of_input_data, "input_ids_data.txt"), mode="rb")
    f2 = open(os.path.join(path_of_input_data, "attention_mask_data.txt"), mode="rb")
    f3 = open(os.path.join(path_of_input_data, "token_type_ids_data.txt"), mode="rb")
    input_ids = pickle.load(f1)
    attention_mask = pickle.load(f2)
    token_type_ids = pickle.load(f3)
    f1.close()
    f2.close()
    f3.close()

    # export onnx
    path_of_output = args.path_of_output
    torch.onnx.export(model, (input_ids, attention_mask, token_type_ids), os.path.join(path_of_output, "model.onnx"),
                      verbose=True,
                      opset_version=12,
                      input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                      dynamic_axes={
                          'input_ids': {0: "batch_size", 1: "token_length"},
                          'attention_mask': {0: "batch_size", 1: "token_length"},
                          'token_type_ids': {0: "batch_size", 1: "token_length"},
                          'answer_start': {0: "batch_size", 1: "token_length"},
                          'answer_end': {0: "batch_size", 1: "token_length"},
                      },
                      output_names=['answer_start', 'answer_end']
                      )


if __name__ == "__main__":
    main()
