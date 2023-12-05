import os
import torch
from mmpretrain.apis import get_model, list_models
from torch.autograd import Variable

class ONNXExporter:
    def __init__(self, source_dir, target_dir, device='cuda:0'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.device = torch.device(device)
        self.dummy_input = Variable(torch.randn(1, 3, 224, 224)).to(self.device)
        self.unsupported_models = []

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def convert_to_onnx(self, model, model_name):
        output_path = os.path.join(self.target_dir, f"{model_name}.onnx")
        model.eval()
        model.to(self.device)
        torch.onnx.export(model,
                          self.dummy_input,
                          output_path,
                          export_params=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        print(f"Exported {model_name} to ONNX format at {output_path}")

    def batch_export(self):
        supported_models = list_models()
        for file_name in os.listdir(self.source_dir):
            if file_name.endswith('.pth'):
                model_path = os.path.join(self.source_dir, file_name)
                model_name, _ = os.path.splitext(file_name)
                
                if model_name not in supported_models:
                    print(f"[WARNING] Model {model_name} is not supported. Skipping.")
                    self.unsupported_models.append(model_name)
                    continue
                
                try:
                    print("[INFO] Starting Export: " + model_name)
                    model = get_model(model_name, pretrained=model_path, device=self.device)
                    self.convert_to_onnx(model, model_name)
                except Exception as e:
                    print(f"[ERROR] Failed to export {model_name}: {e}")
                    self.unsupported_models.append(model_name)

    def report_unsupported_models(self):
        if self.unsupported_models:
            print("[END]")
            print("The following models could not be supported for ONNX export:")
            for model_name in self.unsupported_models:
                print(model_name)
        else:
            print("All models were supported and exported successfully.")

# Usage
if __name__ == '__main__':
    source_dir = './pt'  # Replace with your source directory
    target_dir = './onnx'    # Replace with your target directory
    exporter = ONNXExporter(source_dir, target_dir)
    exporter.batch_export()
    exporter.report_unsupported_models()
