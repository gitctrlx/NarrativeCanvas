import torch
from mmpretrain.apis import get_model
from torch.autograd import Variable

# 确定模型和数据都在同一设备上
device = torch.device('cuda:0')
dummy_input = Variable(torch.randn(1, 3, 224, 224)).to(device)
model_path = '../models/clip-vit-base-p16_openai-pre_3rdparty_in1k.pth'
onnx_output_path = '../onnx/clip-vit-base-p16_openai-pre_3rdparty_in1k.onnx'
model = get_model('vit-base-p16_clip-openai-pre_3rdparty_in1k', pretrained=model_path, device=device)
model.eval()

# 转换为 ONNX 的函数
def convert_to_onnx(model, dummy_input, output_path):
    # 确保模型在推理模式下
    model.eval()
    # 导出模型到ONNX，使用dynamic_axes指定动态批次大小
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      export_params=True, 
                      opset_version=14, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # 动态批次大小
                                    'output': {0: 'batch_size'}})

# 转换为 ONNX
convert_to_onnx(model, dummy_input, onnx_output_path)