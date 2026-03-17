import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from PIL import Image

from tinymobilenet_xs import create_model, quantize_model_post_training
from train import PIEIntentionDataset

CHECKPOINT_PATH = './checkpoints/best_model.pth'
SET02_PKL = './data_cache/sequences/set02_sequences.pkl'
SET05_PKL = './data_cache/sequences/test_sequences.pkl'
PIE_ROOT = '.'
QUANT_CONFIG_PATH = './fpga_weights/quantization_config.json'
RESULTS_PATH = './evaluation_results_set02_set05.json'


class SequenceDataset(Dataset):
    def __init__(self, pkl_file, pie_root='.', max_frames=4):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.image_sequences = data['image']
        self.labels = data['intention_binary']
        self.pie_root = pie_root
        self.max_frames = max_frames

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        image_paths = self.image_sequences[idx][:self.max_frames]
        label = int(self.labels[idx][0][0])

        frames = []
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    filename = Path(path).name
                    for root, _, files in os.walk(os.path.join(self.pie_root, 'images')):
                        if filename in files:
                            path = os.path.join(root, filename)
                            break

                img = Image.open(path).convert('RGB')
                if img.size != (64, 64):
                    img = img.resize((64, 64), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                frames.append(arr)
            except Exception:
                frames.append(np.zeros((64, 64, 3), dtype=np.float32))

        while len(frames) < self.max_frames:
            frames.append(np.zeros((64, 64, 3), dtype=np.float32))

        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return frames, label


def save_quant_config(quant_cfg, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    payload = {
        'weight_scales': {k: float(v) for k, v in quant_cfg.weight_scales.items()},
        'activation_scales': {k: float(v) for k, v in quant_cfg.activation_scales.items()},
        'quantization_bits': 8,
        'note': 'Scales are used to convert FP32 to INT8: int8_value = round(fp32_value * scale)'
    }
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def metrics_from_preds(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        'accuracy': float(100.0 * np.mean(y_true == y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def evaluate_dataset(model, dataset, device, quant_config):
    activation_scales = quant_config.get('activation_scales', {})
    weight_scales = quant_config.get('weight_scales', {})

    gap_scale = float(activation_scales.get('gap', 1.0))
    relu_scale = float(activation_scales.get('relu', 1.0))
    fc2_scale = float(weight_scales.get('fc2', 1.0))

    preds_fp32, preds_int8, gt = [], [], []
    fp32_time, int8_time = 0.0, 0.0

    captured = {'gap': None}

    def gap_hook(module, inp, out):
        captured['gap'] = out.detach()

    hook_handle = model.gap.register_forward_hook(gap_hook)

    with torch.no_grad():
        for i in range(len(dataset)):
            frames, label = dataset[i]
            frames = frames.unsqueeze(0).to(device)

            t0 = time.perf_counter()
            out_fp32 = model(frames)
            fp32_time += (time.perf_counter() - t0)
            pred_fp32 = int(torch.argmax(out_fp32, dim=1).item())

            t1 = time.perf_counter()
            _ = model(frames)
            gap_out = captured['gap']
            if gap_out is not None:
                gap_feat = gap_out.view(1, model.t_frames, 48).mean(dim=1)
                gap_q = torch.clamp(torch.round(gap_feat * gap_scale), -128, 127)
                gap_dq = gap_q / gap_scale

                fc1_out = model.fc1(gap_dq)
                relu_out = torch.relu(fc1_out)
                relu_q = torch.clamp(torch.round(relu_out * relu_scale), -128, 127)
                relu_dq = relu_q / relu_scale

                logits = model.fc2(relu_dq)[0].cpu().numpy()
                logits_q = np.clip(np.round(logits * fc2_scale), -128, 127).astype(np.int8)
                pred_int8 = int(np.argmax(logits_q))
            else:
                pred_int8 = 0

            int8_time += (time.perf_counter() - t1)

            preds_fp32.append(pred_fp32)
            preds_int8.append(pred_int8)
            gt.append(int(label))

            if (i + 1) % 20 == 0:
                print(f'  processed {i + 1}/{len(dataset)}')

    hook_handle.remove()

    m_fp32 = metrics_from_preds(gt, preds_fp32)
    m_int8 = metrics_from_preds(gt, preds_int8)

    agreement = float(100.0 * np.mean(np.array(preds_fp32) == np.array(preds_int8)))

    return {
        'num_samples': len(dataset),
        'class_0': int(np.sum(np.array(gt) == 0)),
        'class_1': int(np.sum(np.array(gt) == 1)),
        'fp32': {
            **m_fp32,
            'avg_inference_ms_per_sample': float((fp32_time / max(1, len(dataset))) * 1000.0),
        },
        'int8': {
            **m_int8,
            'avg_inference_ms_per_sample': float((int8_time / max(1, len(dataset))) * 1000.0),
        },
        'comparison': {
            'accuracy_drop': float(m_fp32['accuracy'] - m_int8['accuracy']),
            'f1_macro_drop': float(m_fp32['f1_macro'] - m_int8['f1_macro']),
            'f1_weighted_drop': float(m_fp32['f1_weighted'] - m_int8['f1_weighted']),
            'prediction_agreement_pct': agreement,
        },
    }


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')
    if not os.path.exists(SET02_PKL):
        raise FileNotFoundError(f'Set02 sequences not found: {SET02_PKL}')
    if not os.path.exists(SET05_PKL):
        raise FileNotFoundError(f'Set05 sequences not found: {SET05_PKL}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n[1/4] Loading model checkpoint...')
    model = create_model(num_classes=2, t_frames=4).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print('\n[2/4] Quantizing model for Python testing (PTQ calibration on set02+set05)...')
    calibration_dataset = ConcatDataset([
        PIEIntentionDataset(SET02_PKL, PIE_ROOT, max_frames=4, split='test'),
        PIEIntentionDataset(SET05_PKL, PIE_ROOT, max_frames=4, split='test'),
    ])
    calibration_loader = DataLoader(calibration_dataset, batch_size=16, shuffle=False, num_workers=0)

    _, quant_cfg = quantize_model_post_training(model, calibration_loader, device=device)
    quant_cfg_json = save_quant_config(quant_cfg, QUANT_CONFIG_PATH)
    print(f'  saved quantization config: {QUANT_CONFIG_PATH}')

    print('\n[3/4] Evaluating FP32 vs INT8 on set02 and set05...')
    set02_ds = SequenceDataset(SET02_PKL, PIE_ROOT, max_frames=4)
    set05_ds = SequenceDataset(SET05_PKL, PIE_ROOT, max_frames=4)
    combined_ds = ConcatDataset([set02_ds, set05_ds])

    print(' evaluating set02')
    set02_result = evaluate_dataset(model, set02_ds, device, quant_cfg_json)

    print(' evaluating set05')
    set05_result = evaluate_dataset(model, set05_ds, device, quant_cfg_json)

    print(' evaluating combined')
    combined_result = evaluate_dataset(model, combined_ds, device, quant_cfg_json)

    print('\n[4/4] Saving results...')
    results = {
        'checkpoint': CHECKPOINT_PATH,
        'quant_config': QUANT_CONFIG_PATH,
        'set02': set02_result,
        'set05': set05_result,
        'combined': combined_result,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f' saved evaluation: {RESULTS_PATH}')

    for name, r in [('set02', set02_result), ('set05', set05_result), ('combined', combined_result)]:
        print('\n' + '=' * 72)
        print(name.upper())
        print('=' * 72)
        print(f"samples: {r['num_samples']} | class0: {r['class_0']} | class1: {r['class_1']}")
        print(f"FP32 acc: {r['fp32']['accuracy']:.2f}% | INT8 acc: {r['int8']['accuracy']:.2f}% | drop: {r['comparison']['accuracy_drop']:.2f}%")
        print(f"FP32 macro-F1: {r['fp32']['f1_macro']:.4f} | INT8 macro-F1: {r['int8']['f1_macro']:.4f} | drop: {r['comparison']['f1_macro_drop']:.4f}")
        print(f"agreement: {r['comparison']['prediction_agreement_pct']:.2f}%")
        print(f"latency ms/sample -> FP32: {r['fp32']['avg_inference_ms_per_sample']:.3f}, INT8(sim): {r['int8']['avg_inference_ms_per_sample']:.3f}")


if __name__ == '__main__':
    main()
