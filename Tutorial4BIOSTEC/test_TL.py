# from NeuralLib.architectures import GRUseq2seq
# from NeuralLib.config import DATASETS_ECG_G  # peaks dataset
# from NeuralLib.model_hub import (
#     TLModel, TLFactory, 
#     list_production_models, 
#     build_layer_mapping_from_pretrained, 
#     build_tl_arch_params_from_pretrained,
#     build_freeze_phase1,
#     build_freeze_phase2)
# import os


# PRETRAINED_MODEL_NAME = "ECGPeakDetector"

# X = os.path.join(DATASETS_ECG_G, "x")
# Y_BIN = os.path.join(DATASETS_ECG_G, "y_bin")

# factory = TLFactory()

# # Optional: list models in the HF collection
# list_production_models()

# # Load Production Model
# factory.load_production_model(model_name=PRETRAINED_MODEL_NAME)
# prod_model = factory.models[PRETRAINED_MODEL_NAME]

# print("\nPretrained GRU model (encoder):")
# print(prod_model.model)

# # TODO: Build better TL model bidirectional GRU parameters
# # example: bidirectional gru in last layer should be False for regression tasks.
# # Other models (seq2one, ect...)
# arch_params, reuse_n = build_tl_arch_params_from_pretrained(
#     prod_model=prod_model,
#     reuse_n_gru_layers=1,
#     extra_hid_dims=[64, 128],
#     learning_rate=1e-3,
#     model_name_suffix="_TL_PeakDetector",
#     task="regression",
#     num_classes=1,
#     multi_label=False,    # BCEWithLogits for binary peaks
#     fc_out_bool=False,
# )

# print("\nTL architecture hyperparameters:")
# print(arch_params)

# tl_model = TLModel(GRUseq2seq, **arch_params)
# print(f"TL_model structure{tl_model}")

# print("\nTLModel architecture:")
# print(tl_model.model)
# # Print all keys (TLModel) in the state_dict
# print("\nTLModel Keys:")
# for key in tl_model.state_dict().keys():
#     print(key)

# layer_mapping = build_layer_mapping_from_pretrained(prod_model, reuse_n)

# print("\nReused GRU layers:")
# for k in layer_mapping.keys():
#     print("  ", k)

# # Automatic freeze_layers setting
# # freeze_layers, unfreeze_layers = build_freeze_phase1(tl_model, reuse_n)
# freeze_layers, unfreeze_layers = build_freeze_phase2(tl_model, reuse_n, strategy="all")

# factory.configure_tl_model(
#     tl_model=tl_model,
#     layer_mapping=layer_mapping,
#     freeze_layers=freeze_layers,
#     unfreeze_layers=unfreeze_layers,
# )

# train_params = {
#     "path_x": X,
#     "path_y": Y_BIN,
#     "epochs": 5,
#     "batch_size": 8,
#     "patience": 2,
#     "dataset_name": "private_gib01",
#     "trained_for": "fine-tuning peak detection",
#     "all_samples": False,
#     "samples": 50,
#     "gpu_id": None,
#     "enable_tensorboard": True,
# }

# print("\n Starting TL training...")
# tl_model.train_tl(**train_params)
# print(" TL training finished.")

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import NeuralLib.model_hub as mh
from NeuralLib.architectures import post_process_peaks_binary

# ----------------- 配置 -----------------
INPUT_DIR = "/Users/groupies/Documents/NOVAProjects/NeuLib/mitbih_peak_dataset/ALL_group_preprocessed/sr360/x/train"          # 输入 .npy 信号文件夹
OUTPUT_DIR = "/Users/groupies/Documents/NOVAProjects/NeuLib/mitbih_peak_dataset/ALL_group_preprocessed/sr360/denoised_peaks"         # 输出图片文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPU_ID = None  # 没 GPU 就用 None

# ----------------- 加载预训练模型 -----------------
print("Loading pretrained models...")
denoiser = mh.ProductionModel(model_name="ECGDenoiser")
peak_detector = mh.ProductionModel(model_name="ECGPeakDetector")
print("Models ready.")

# ----------------- 工具函数 -----------------
import torch
import numpy as np
from NeuralLib.architectures import post_process_peaks_binary

def predict_peaks(model, signal, gpu_id=None, threshold=0.5):
    # 1) 保证输入给 model.predict 的是 1D numpy
    signal_np = np.array(signal).reshape(-1)

    raw_pred = model.predict(
        X=signal_np,
        post_process_fn=post_process_peaks_binary,  # convert logits to peak indices
        threshold=0.5,
        filter_peaks=True,
    )

    return raw_pred, signal_np

def process_file(path):
    fname = os.path.basename(path)
    signal = np.load(path)

    # 确保是 1D
    if signal.ndim > 1:
        signal = signal.squeeze()

    s_min = signal.min()
    s_max = signal.max()
    if s_min == s_max:
        signal = np.zeros_like(signal)
    else:
        signal = (signal - s_min) / (s_max - s_min)
        
    # 1) 去噪
    denoised = denoiser.predict(
        X=signal,
        gpu_id=GPU_ID,
        post_process_fn=None,
    )
    denoised = np.array(denoised).reshape(-1)

    # 2) 原始信号 peak detection
    peaks_raw, _ = predict_peaks(peak_detector, signal, gpu_id=GPU_ID)

    # 3) 去噪信号 peak detection
    peaks_denoised, _ = predict_peaks(peak_detector, denoised, gpu_id=GPU_ID)

    # 4) 绘图并保存
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(signal))

    # 上：原始信号 + peaks
    ax = axes[0]
    ax.plot(t, signal, color="#1f77b4", linewidth=1.0, label="Raw ECG")
    ax.scatter(
        peaks_raw,
        signal[peaks_raw],
        color="#d62728",
        s=20,
        label="Peaks (raw)",
        zorder=3,
    )
    ax.set_ylabel("Amplitude")
    ax.set_title("Raw ECG with peak detector output")
    ax.legend(loc="upper right", frameon=False)

    # 下：去噪信号 + peaks
    ax = axes[1]
    ax.plot(t, denoised, color="#2ca02c", linewidth=1.0, label="Denoised ECG")
    ax.scatter(
        peaks_denoised,
        denoised[peaks_denoised],
        color="#ff7f0e",
        s=20,
        label="Peaks (denoised)",
        zorder=3,
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title("Denoised ECG with peak detector output")
    ax.legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".npy", "_denoise_peaks.png"))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ----------------- 主循环 -----------------
if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npy"))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    for f in files:
        process_file(os.path.join(INPUT_DIR, f))
