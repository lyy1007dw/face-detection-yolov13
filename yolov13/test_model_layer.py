import torch
import sys


def get_layers(model):
    """
    递归获取 YOLO 真正的层 (nn.ModuleList)
    """
    if hasattr(model, 'model'):
        return get_layers(model.model)
    return model


def debug_yolo_layers(model, input_size=(1, 3, 1280, 1280)):
    """
    支持 YOLOv13 的逐层调试（含 from 结构）
    """
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)

    print("=" * 60)
    print(f"输入尺寸: {x.shape}")
    print("=" * 60)

    layers = get_layers(model)

    outputs = []   # 保存每一层输出
    features = {}

    for i, layer in enumerate(layers):
        try:
            # ===== 处理输入 (from结构) =====
            if hasattr(layer, 'f'):
                f = layer.f
                if f == -1:
                    inp = x if i == 0 else outputs[-1]
                elif isinstance(f, int):
                    inp = outputs[f]
                else:
                    inp = [outputs[j] for j in f]
            else:
                inp = outputs[-1] if outputs else x

            # ===== 输入尺寸 =====
            if isinstance(inp, torch.Tensor):
                in_shape = tuple(inp.shape)
            elif isinstance(inp, list):
                in_shape = [tuple(t.shape) for t in inp]
            else:
                in_shape = str(type(inp))

            # ===== 前向传播 =====
            out = layer(inp)

            # ===== 输出尺寸 =====
            if isinstance(out, torch.Tensor):
                out_shape = tuple(out.shape)
            elif isinstance(out, (list, tuple)):
                out_shape = [tuple(t.shape) for t in out if isinstance(t, torch.Tensor)]
            else:
                out_shape = str(type(out))

            # ===== 打印 =====
            print(f"层 {i:2d}: {layer.__class__.__name__:<25}")
            print(f"  from: {getattr(layer, 'f', 'N/A')}")
            print(f"  输入: {in_shape}")
            print(f"  输出: {out_shape}")
            print("-" * 50)

            outputs.append(out)
            features[i] = out

        except Exception as e:
            print(f"\n层 {i} 执行失败: {layer.__class__.__name__}")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            break

    return features


# ================= 主函数 =================
if __name__ == "__main__":

    # ===== 加载模型 =====
    try:
        print("尝试加载 best.pt ...")
        ckpt = torch.load('best.pt', map_location='cpu')

        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = ckpt['model']
        else:
            model = ckpt

    except Exception as e:
        print("加载 pt 失败，尝试使用 YOLO API")
        try:
            from ultralytics import YOLO
            model = YOLO('ultralytics/cfg/models/v13/yolov13p2.yaml').model
        except:
            print("无法加载模型")
            sys.exit(1)

    # ===== 设置模型 =====
    model = model.float().eval()

    # ===== 测试尺寸 =====
    test_sizes = [
        (1, 3, 1280, 1280),
    ]

    for size in test_sizes:
        print(f"\n\n测试输入尺寸: {size}")
        print("=" * 60)

        features = debug_yolo_layers(model, input_size=size)

        # ===== 打印关键层 =====
        print("\n前32层输出尺寸:")
        for i in range(32):
            if i in features:
                f = features[i]
                if isinstance(f, torch.Tensor):
                    print(f"层 {i:2d}: {tuple(f.shape)}")
                else:
                    print(f"层 {i:2d}: 非Tensor")