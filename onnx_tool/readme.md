# usage
This is the command to convert the PyTorch model ("du2021/384_bert/checkpoint-66") to the ONNX model (./model.onnx).
```
python onnx_export.py --path_of_pt_model "du2021/384_bert/checkpoint-66" --path_of_input_data "resources/for_onnx_export" --path_of_output "."
```