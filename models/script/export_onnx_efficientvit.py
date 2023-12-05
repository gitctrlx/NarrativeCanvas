# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse

import torch

from efficientvit.apps.utils import export_onnx
from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.models.utils import val2tuple
from efficientvit.seg_model_zoo import create_seg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", type=str, default="./models/b3-r224.onnx")
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "seg"])
    parser.add_argument("--dataset", type=str, default="none", choices=["ade20k", "cityscapes"])
    parser.add_argument("--model", type=str, default="b3")
    parser.add_argument("--resolution", type=int, nargs="+", default=224)
    parser.add_argument("--bs", help="batch size", type=int, default=16)
    parser.add_argument("--op_set", type=int, default=11)

    args = parser.parse_args()

    resolution = val2tuple(args.resolution, 2)
    if args.task == "cls":
        model = create_cls_model(
            name=args.model,
            pretrained=True,
            weight_url = "./models/b3-r224.pt"
        )
    elif args.task == "seg":
        model = create_seg_model(
            name=args.model,
            dataset=args.dataset,
            pretrained=False,
        )
    else:
        raise NotImplementedError

    # dummy_input = torch.rand((args.bs, 3, *resolution))
    dummy_input = torch.randn(1, 3, *resolution)
    # export_onnx(model, args.export_path, dummy_input, simplify=True, opset=args.op_set)
    torch.onnx.export(
        model, 
        dummy_input, 
        args.export_path,
        opset_version=args.op_set, 
        do_constant_folding=True,
        input_names=['input'],  # 为便于参考，给输入张量命名
        output_names=['output'],  # 为便于参考，给输出张量命名
        dynamic_axes={'input' : {0 : 'batch_size'},  # 第一个维度是批次大小
                      'output' : {0 : 'batch_size'}}  # 如果你的模型输出也有动态批次大小
    )


if __name__ == "__main__":
    main()
