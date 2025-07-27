import copy
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def apply_quantization(model, device='cuda', model_quantization='fp32'):

    if model_quantization == 'fp16':
        quantized_model = copy.deepcopy(model).half().to(device)
    elif model_quantization == 'int8':
        quantized_model = copy.deepcopy(model).to('cpu')
        quantized_model.eval()

        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model,
            {
                torch.nn.Linear,
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            }, 
            dtype=torch.qint8
        )
    else:
        raise ValueError(f"Unsupported model quantization: {model_quantization}")

    return quantized_model

def get_model_size(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb

def evaluate_quantization(model, data_loader, device='cuda', model_quantization='fp32'):
    model.eval()

    all_preds = []
    all_labels = []

    model_device = 'cpu' if model_quantization == 'int8' else device

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(tqdm(data_loader, desc=f'Evaluating {model_quantization}')):

            inputs = inputs.to(model_device)
            labels = labels.to(model_device)

            if model_quantization == 'fp16':
                inputs = inputs.half()

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    mem_size = get_model_size(model)

    return accuracy, mem_size