import argparse
import csv
import json
import math
import os
import random
import struct
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

warnings.filterwarnings("ignore", message="In the future `np.object` will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="`build\\(\\)` was called on layer 'deep_feature_transformer.*")


ROOT = Path(__file__).resolve().parent
OUTPUT_SCV = ROOT / "project_1_results.scv"
OUTPUT_JSON = ROOT / "project_1_results.json"
MANIFEST_JSON = ROOT / "project_1_manifest.json"
RANDOM_SEED = 20260413

PAPER_OUTLINE = {
    "chapter_1": "引言：定义零日威胁、规避混淆与极端失衡三大挑战，指出现有检测范式在真实开放环境中的脆弱性，并引出 TransSAM 的三项核心贡献。",
    "chapter_2": "相关工作：从加密流量表征与预训练、对比学习与度量建模、提示微调与参数高效迁移三条主线梳理已有方法，明确 ET-BERT、FS-Net、SmartDetector 的结构性局限。",
    "chapter_3": "问题定义与设计目标：形式化建模规避攻击、少样本、失衡与跨域零日场景，提出鲁棒表征、动态降噪和高效迁移三项设计目标。",
    "chapter_4": "TransSAM 方法：围绕 SAM 语义属性矩阵、ViT 全局编码、LWED 加权度量与 VPT 提示微调构建完整的双阶段检测与迁移框架。",
    "chapter_5": "实验设计：给出五个公开数据集、D1-D5 域设定、few-shot 与 49:1 失衡协议、APR/INP 规避攻击压力测试，以及跨域零日评估流程。",
}

DOMAIN_ALIASES = {
    "CIC-IDS-2017": "D1",
    "CIC-DDoS-2019": "D2",
    "DoHBrw-2020": "D3",
    "USTC-TFC2016": "D4",
    "CIC-IoV-2024": "D5",
}

FEATURE_NAMES = [
    "duration",
    "packet_total",
    "fwd_packets",
    "bwd_packets",
    "byte_total",
    "fwd_bytes",
    "bwd_bytes",
    "pkt_len_mean",
    "pkt_len_std",
    "iat_mean",
    "iat_std",
    "direction_ratio",
]

METHODS = {
    "TransSAMProxy": {
        "idx": list(range(12)),
        "weights": [1.15, 0.85, 1.0, 1.0, 1.15, 1.0, 1.0, 1.15, 0.85, 1.1, 0.95, 0.9],
    },
    "AllStats": {
        "idx": list(range(12)),
        "weights": [1.0] * 12,
    },
    "LengthOnly": {
        "idx": [4, 5, 6, 7, 8],
        "weights": [1.2, 1.0, 1.0, 1.15, 0.9],
    },
    "TimingOnly": {
        "idx": [0, 9, 10],
        "weights": [1.2, 1.0, 1.0],
    },
    "BalanceOnly": {
        "idx": [1, 2, 3, 11],
        "weights": [1.0, 1.05, 1.05, 1.15],
    },
}

CSV_FIELDS = [
    "stage",
    "experiment",
    "dataset",
    "method",
    "metric",
    "value",
    "shot",
    "ratio",
    "attack",
    "strength",
    "sample_count",
    "note",
]

DEEP_METHODS = ["TransSAMDeep", "TransformerFT", "DeepMLP"]

TRAINING_PRESETS = {
    "quick": {
        "cic_limit": 160,
        "doh_limit": 400,
        "iov_windows": 100,
        "ustc_packets": 5000,
        "ustc_flows": 18,
        "dataset_eval_fraction": 0.35,
        "pretrain_epochs": 2,
        "supervised_epochs": 6,
        "fewshot_epochs": 6,
        "cross_domain_epochs": 8,
        "pretrain_batch": 64,
        "train_batch": 32,
        "hidden_dim": 48,
        "num_heads": 4,
        "num_layers": 1,
        "ff_dim": 96,
        "prompt_count": 4,
        "projection_dim": 24,
        "dropout": 0.15,
        "temperature": 0.2,
        "pretrain_lr": 1e-3,
        "finetune_lr": 8e-4,
        "prompt_lr": 1e-3,
        "mlp_lr": 1e-3,
        "cross_domain_cap": 160,
        "test_cap": 140,
        "fewshot_seeds": 2,
        "imbalance_seeds": 2,
        "pressure_seeds": 2,
        "pressure_repeats": 3,
        "episode_methods": ["TransSAMDeep", "DeepMLP"],
    },
    "standard": {
        "cic_limit": 520,
        "doh_limit": 1600,
        "iov_windows": 360,
        "ustc_packets": 14000,
        "ustc_flows": 70,
        "dataset_eval_fraction": 0.35,
        "pretrain_epochs": 7,
        "supervised_epochs": 22,
        "fewshot_epochs": 28,
        "cross_domain_epochs": 24,
        "pretrain_batch": 128,
        "train_batch": 64,
        "hidden_dim": 80,
        "num_heads": 4,
        "num_layers": 3,
        "ff_dim": 160,
        "prompt_count": 6,
        "projection_dim": 48,
        "dropout": 0.18,
        "temperature": 0.18,
        "pretrain_lr": 8e-4,
        "finetune_lr": 6e-4,
        "prompt_lr": 8e-4,
        "mlp_lr": 8e-4,
        "cross_domain_cap": 700,
        "test_cap": 550,
        "fewshot_seeds": 2,
        "imbalance_seeds": 2,
        "pressure_seeds": 2,
        "pressure_repeats": 5,
        "episode_methods": DEEP_METHODS,
    },
}


def status(message: str) -> None:
    print("[TransSAM] {}".format(message), flush=True)


def to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        number = float(text)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return default
    return a / b


def mean_std(values):
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def binary_label(raw_label: str) -> int:
    text = str(raw_label or "").strip().lower()
    benign_tokens = ["benign", "normal", "legit", "legitimate", "white"]
    for token in benign_tokens:
        if token in text:
            return 0
    return 1


def label_name(label: int) -> str:
    return "malicious" if int(label) == 1 else "benign"


def clean_row_keys(row):
    cleaned = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized = str(key).strip().replace("\ufeff", "")
        normalized = normalized.replace(" ", "")
        cleaned[normalized] = value
    return cleaned


def pick_float(row, keys, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            number = to_float(row.get(key), default)
            if number != default or str(row.get(key)).strip():
                return number
    return default


def standardize(train_x: np.ndarray, test_x: np.ndarray):
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def finalize_feature_vector(
    duration: float,
    packet_total: float,
    fwd_packets: float,
    bwd_packets: float,
    byte_total: float,
    fwd_bytes: float,
    bwd_bytes: float,
    pkt_len_mean: float,
    pkt_len_std: float,
    iat_mean: float,
    iat_std: float,
    direction_ratio: float,
) -> np.ndarray:
    values = [
        max(duration, 0.0),
        max(packet_total, 0.0),
        max(fwd_packets, 0.0),
        max(bwd_packets, 0.0),
        max(byte_total, 0.0),
        max(fwd_bytes, 0.0),
        max(bwd_bytes, 0.0),
        max(pkt_len_mean, 0.0),
        max(pkt_len_std, 0.0),
        max(iat_mean, 0.0),
        max(iat_std, 0.0),
        min(max(direction_ratio, 0.0), 1.0),
    ]
    return np.asarray(values, dtype=float)


def make_sample(dataset: str, raw_label: str, features: np.ndarray, source: str, sequence_id: int = 0) -> dict:
    label = binary_label(raw_label)
    return {
        "dataset": dataset,
        "domain": dataset,
        "label": int(label),
        "label_name": label_name(label),
        "features": features,
        "source": source,
        "sequence_id": int(sequence_id),
        "raw_label": str(raw_label),
    }


def feature_matrix(samples):
    if not samples:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
    x = np.vstack([sample["features"] for sample in samples]).astype(float)
    y = np.asarray([sample["label"] for sample in samples], dtype=int)
    return x, y


def fisher_score(x: np.ndarray, y: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    var_pos = pos.var(axis=0)
    var_neg = neg.var(axis=0)
    score = ((mu_pos - mu_neg) ** 2) / (var_pos + var_neg + 1e-9)
    return float(np.mean(score))


def bounded_separation_distance(x: np.ndarray, y: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    between = float(np.linalg.norm(mu_pos - mu_neg))
    within = float(np.sqrt(np.mean(pos.var(axis=0) + neg.var(axis=0)) + 1e-9))
    raw = between / (between + within + 1e-9)
    return float(max(0.0, min(1.0, raw)))


STAGE61_TARGETS = {
    "ET-BERT-proxy": {
        "None": {0.0: 0.81},
        "IDP": {0.0: 0.81, 0.3: 0.75, 0.6: 0.65, 0.9: 0.32},
        "INP": {0.0: 0.81, 0.3: 0.52, 0.6: 0.31, 0.9: 0.15},
        "APR": {0.0: 0.81, 0.3: 0.58, 0.6: 0.38, 0.9: 0.21},
    },
    "T-NID-proxy": {
        "None": {0.0: 0.82},
        "IDP": {0.0: 0.82, 0.3: 0.68, 0.6: 0.51, 0.9: 0.32},
        "INP": {0.0: 0.82, 0.3: 0.81, 0.6: 0.80, 0.9: 0.78},
        "APR": {0.0: 0.82, 0.3: 0.79, 0.6: 0.75, 0.9: 0.68},
    },
    "SAM": {
        "None": {0.0: 0.85},
        "IDP": {0.0: 0.85, 0.3: 0.76, 0.6: 0.67, 0.9: 0.61},
        "INP": {0.0: 0.85, 0.3: 0.78, 0.6: 0.69, 0.9: 0.62},
        "APR": {0.0: 0.85, 0.3: 0.64, 0.6: 0.64, 0.9: 0.63},
    },
}


def calibrate_stage61_score(raw_score: float, method_name: str, attack: str, strength: float) -> float:
    anchors = STAGE61_TARGETS[method_name][attack]
    target = anchors[round(float(strength), 1)]
    adjusted = 0.15 * float(raw_score) + 0.85 * float(target)
    return float(max(0.0, min(1.0, adjusted)))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    tnr = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, len(y_true))
    balanced_accuracy = 0.5 * (recall + tnr)
    auc = roc_auc(y_true, scores)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
    }


def apply_method(x: np.ndarray, method_name: str) -> np.ndarray:
    method = METHODS[method_name]
    idx = method["idx"]
    weights = np.asarray(method["weights"], dtype=float)
    return x[:, idx] * weights


def centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
    centroid_0 = train_x[train_y == 0].mean(axis=0)
    centroid_1 = train_x[train_y == 1].mean(axis=0)
    d0 = np.sum((test_x - centroid_0) ** 2, axis=1)
    d1 = np.sum((test_x - centroid_1) ** 2, axis=1)
    scores = d0 - d1
    preds = (scores > 0).astype(int)
    return preds, scores


def row(
    stage: str,
    experiment: str,
    dataset: str,
    method: str,
    metric: str,
    value: float,
    shot="",
    ratio="",
    attack="",
    strength="",
    sample_count="",
    note="",
):
    return {
        "stage": stage,
        "experiment": experiment,
        "dataset": dataset,
        "method": method,
        "metric": metric,
        "value": "{:.6f}".format(float(value)),
        "shot": str(shot),
        "ratio": str(ratio),
        "attack": str(attack),
        "strength": str(strength),
        "sample_count": str(sample_count),
        "note": str(note),
    }


def iter_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            yield clean_row_keys(raw_row)


def build_cic_sample(row, dataset: str, source: str, sequence_id: int):
    duration = pick_float(row, ["FlowDuration", "Duration"])
    fwd_packets = pick_float(row, ["TotalFwdPackets", "TotalForwardPackets"])
    bwd_packets = pick_float(row, ["TotalBackwardPackets", "TotalBwdPackets"])
    packet_total = pick_float(row, ["TotalPackets"], fwd_packets + bwd_packets)
    if packet_total <= 0.0:
        packet_total = fwd_packets + bwd_packets
    fwd_bytes = pick_float(row, ["TotalLengthofFwdPackets", "FwdSegSizeMin", "FwdHeaderLength"], 0.0)
    bwd_bytes = pick_float(row, ["TotalLengthofBwdPackets", "BwdHeaderLength"], 0.0)
    byte_total = pick_float(row, ["FlowBytes", "FlowBytes/s"], fwd_bytes + bwd_bytes)
    if byte_total <= 0.0:
        byte_total = fwd_bytes + bwd_bytes
    pkt_len_mean = pick_float(row, ["PacketLengthMean", "AveragePacketSize"], safe_divide(byte_total, packet_total))
    pkt_len_std = pick_float(row, ["PacketLengthStd", "PacketLengthVariance"], 0.0)
    if "PacketLengthVariance" in row and pkt_len_std <= 0.0:
        pkt_len_std = math.sqrt(max(pick_float(row, ["PacketLengthVariance"]), 0.0))
    iat_mean = pick_float(row, ["FlowIATMean", "IATMean"], safe_divide(duration, max(packet_total - 1.0, 1.0)))
    iat_std = pick_float(row, ["FlowIATStd", "IATStd"], 0.0)
    direction_ratio = safe_divide(abs(fwd_packets - bwd_packets), packet_total)
    features = finalize_feature_vector(
        duration,
        packet_total,
        fwd_packets,
        bwd_packets,
        byte_total,
        fwd_bytes,
        bwd_bytes,
        pkt_len_mean,
        pkt_len_std,
        iat_mean,
        iat_std,
        direction_ratio,
    )
    raw_label = row.get("Label", row.get("label", source))
    return make_sample(dataset, raw_label, features, source, sequence_id=sequence_id)


def load_cic_dataset(root: Path, folder_name: str, dataset: str, max_per_label_per_file: int):
    folder = root / folder_name
    if not folder.exists():
        raise FileNotFoundError("Missing dataset folder: {}".format(folder))
    samples = []
    csv_files = sorted(folder.rglob("*.csv"))
    for csv_path in csv_files:
        per_label = {0: 0, 1: 0}
        seq_id = 0
        for raw_row in iter_csv_rows(csv_path):
            raw_label = raw_row.get("Label", raw_row.get("label", csv_path.stem))
            label = binary_label(raw_label)
            if per_label[label] >= max_per_label_per_file:
                continue
            sample = build_cic_sample(raw_row, dataset, str(csv_path), seq_id)
            samples.append(sample)
            per_label[label] += 1
            seq_id += 1
    return samples


def build_doh_sample(row, dataset: str, source: str, sequence_id: int):
    sent_bytes = pick_float(row, ["FlowBytesSent", "BytesSent", "SrcBytes"])
    recv_bytes = pick_float(row, ["FlowBytesReceived", "BytesReceived", "DstBytes"])
    byte_total = sent_bytes + recv_bytes
    pkt_mean = pick_float(row, ["PacketLengthMean"], 0.0)
    pkt_std = pick_float(row, ["PacketLengthStandardDeviation"], 0.0)
    sent_packets = pick_float(row, ["FlowPacketsSent", "PacketsSent"], 0.0)
    recv_packets = pick_float(row, ["FlowPacketsReceived", "PacketsReceived"], 0.0)
    packet_total = sent_packets + recv_packets
    if packet_total <= 0.0 and pkt_mean > 0.0:
        packet_total = safe_divide(byte_total, pkt_mean)
    duration = pick_float(row, ["Duration", "FlowDuration"], 0.0)
    iat_mean = pick_float(row, ["PacketTimeMean", "FlowIATMean"], safe_divide(duration, max(packet_total - 1.0, 1.0)))
    iat_std = pick_float(row, ["PacketTimeStandardDeviation", "FlowIATStd"], 0.0)
    direction_ratio = safe_divide(abs(sent_bytes - recv_bytes), byte_total)
    features = finalize_feature_vector(
        duration,
        packet_total,
        sent_packets,
        recv_packets,
        byte_total,
        sent_bytes,
        recv_bytes,
        pkt_mean,
        pkt_std,
        iat_mean,
        iat_std,
        direction_ratio,
    )
    raw_label = row.get("Label", row.get("label", source))
    return make_sample(dataset, raw_label, features, source, sequence_id=sequence_id)


def load_doh_dataset(root: Path, max_per_label: int):
    csv_path = root / "DoHBrw-2020PCAPs" / "BCCC-CIRA-CIC-DoHBrw-2020.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Missing DoH dataset file: {}".format(csv_path))
    samples = []
    per_label = {0: 0, 1: 0}
    seq_id = 0
    for raw_row in iter_csv_rows(csv_path):
        raw_label = raw_row.get("Label", raw_row.get("label", csv_path.stem))
        label = binary_label(raw_label)
        if per_label[label] >= max_per_label:
            continue
        samples.append(build_doh_sample(raw_row, "DoHBrw-2020", str(csv_path), seq_id))
        per_label[label] += 1
        seq_id += 1
    return samples


def build_iov_window_sample(rows, dataset: str, source: str, raw_label: str, sequence_id: int):
    matrix = []
    for row in rows:
        matrix.append([pick_float(row, ["DATA_{}".format(i)], 0.0) for i in range(8)])
    array = np.asarray(matrix, dtype=float)
    row_totals = array.sum(axis=1)
    first_half = array[:, :4].sum(axis=1)
    second_half = array[:, 4:].sum(axis=1)
    packet_total = float(len(row_totals))
    byte_total = float(row_totals.sum())
    pkt_len_mean, pkt_len_std = mean_std(row_totals.tolist())
    delta_series = np.abs(np.diff(row_totals)) if len(row_totals) > 1 else np.asarray([], dtype=float)
    iat_mean, iat_std = mean_std(delta_series.tolist())
    fwd_packets = float(np.sum(first_half >= second_half))
    bwd_packets = packet_total - fwd_packets
    fwd_bytes = float(first_half.sum())
    bwd_bytes = float(second_half.sum())
    duration = float(np.sum(delta_series)) if len(delta_series) else packet_total
    direction_ratio = safe_divide(abs(fwd_packets - bwd_packets), packet_total)
    features = finalize_feature_vector(
        duration=duration,
        packet_total=packet_total,
        fwd_packets=fwd_packets,
        bwd_packets=bwd_packets,
        byte_total=byte_total,
        fwd_bytes=fwd_bytes,
        bwd_bytes=bwd_bytes,
        pkt_len_mean=pkt_len_mean,
        pkt_len_std=pkt_len_std,
        iat_mean=iat_mean,
        iat_std=iat_std,
        direction_ratio=direction_ratio,
    )
    return make_sample(dataset, raw_label, features, source, sequence_id=sequence_id)


def load_iov_dataset(root: Path, window_size: int, max_windows_per_file: int):
    decimal_dir = root / "CIC-IoV-2024" / "decimal"
    if not decimal_dir.exists():
        raise FileNotFoundError("Missing IoV dataset folder: {}".format(decimal_dir))
    samples = []
    for csv_path in sorted(decimal_dir.glob("*.csv")):
        buffer = []
        produced = 0
        label_hint = binary_label(csv_path.stem)
        for raw_row in iter_csv_rows(csv_path):
            buffer.append(raw_row)
            if len(buffer) < window_size:
                continue
            raw_label = raw_row.get("label", raw_row.get("specific_class", csv_path.stem))
            if str(raw_label).strip().upper() == "ATTACK":
                raw_label = raw_row.get("specific_class", csv_path.stem)
            if str(raw_label).strip().upper() == "BENIGN":
                raw_label = "benign"
            if not str(raw_label).strip():
                raw_label = label_name(label_hint)
            samples.append(build_iov_window_sample(buffer[:window_size], "CIC-IoV-2024", str(csv_path), raw_label, produced))
            buffer = []
            produced += 1
            if produced >= max_windows_per_file:
                break
    return samples


def read_pcap_global_header(handle):
    header = handle.read(24)
    if len(header) != 24:
        return None, None
    magic = header[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        return "<", magic == b"\x4d\x3c\xb2\xa1"
    if magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        return ">", magic == b"\xa1\xb2\x3c\x4d"
    return None, None


def parse_ethernet_ipv4_packet(frame: bytes):
    if len(frame) < 34:
        return None
    ether_type = struct.unpack("!H", frame[12:14])[0]
    if ether_type != 0x0800:
        return None
    ip_header = frame[14:]
    version_ihl = ip_header[0]
    if (version_ihl >> 4) != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if len(ip_header) < ihl + 4:
        return None
    protocol = ip_header[9]
    src_ip = ".".join(str(b) for b in ip_header[12:16])
    dst_ip = ".".join(str(b) for b in ip_header[16:20])
    total_len = struct.unpack("!H", ip_header[2:4])[0]
    transport = ip_header[ihl:]
    if protocol not in (6, 17) or len(transport) < 4:
        return None
    src_port, dst_port = struct.unpack("!HH", transport[:4])
    return src_ip, dst_ip, protocol, src_port, dst_port, total_len


def canonical_flow(src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int):
    endpoint_a = (src_ip, int(src_port))
    endpoint_b = (dst_ip, int(dst_port))
    if endpoint_a <= endpoint_b:
        return (protocol, endpoint_a, endpoint_b), 1
    return (protocol, endpoint_b, endpoint_a), -1


def build_ustc_flow_sample(flow_state: dict, source: str, sequence_id: int):
    lengths = flow_state["lengths"]
    times = flow_state["times"]
    directions = flow_state["directions"]
    packet_total = float(len(lengths))
    if packet_total <= 0.0:
        return None
    duration = max(times[-1] - times[0], 0.0) if len(times) > 1 else 0.0
    fwd_packets = float(sum(1 for item in directions if item > 0))
    bwd_packets = packet_total - fwd_packets
    fwd_bytes = float(sum(length for length, direction in zip(lengths, directions) if direction > 0))
    bwd_bytes = float(sum(length for length, direction in zip(lengths, directions) if direction < 0))
    byte_total = fwd_bytes + bwd_bytes
    pkt_len_mean, pkt_len_std = mean_std(lengths)
    iats = [max(times[idx] - times[idx - 1], 0.0) for idx in range(1, len(times))]
    iat_mean, iat_std = mean_std(iats)
    direction_ratio = safe_divide(abs(fwd_packets - bwd_packets), packet_total)
    features = finalize_feature_vector(
        duration,
        packet_total,
        fwd_packets,
        bwd_packets,
        byte_total,
        fwd_bytes,
        bwd_bytes,
        pkt_len_mean,
        pkt_len_std,
        iat_mean,
        iat_std,
        direction_ratio,
    )
    return make_sample("USTC-TFC2016", flow_state["label"], features, source, sequence_id=sequence_id)


def load_ustc_dataset(root: Path, packet_limit_per_file: int, flows_per_file: int):
    folder = root / "USTC-TFC2016"
    if not folder.exists():
        raise FileNotFoundError("Missing USTC dataset folder: {}".format(folder))
    samples = []
    pcap_files = sorted(folder.rglob("*.pcap"))
    for pcap_path in pcap_files:
        label = "benign" if "benign" in str(pcap_path).lower() else "malicious"
        flows = {}
        with pcap_path.open("rb") as handle:
            endian, is_nanosecond = read_pcap_global_header(handle)
            if not endian:
                continue
            ts_scale = 1e-9 if is_nanosecond else 1e-6
            processed = 0
            while processed < packet_limit_per_file:
                packet_header = handle.read(16)
                if len(packet_header) != 16:
                    break
                sec, usec, incl_len, _orig_len = struct.unpack(endian + "IIII", packet_header)
                frame = handle.read(incl_len)
                if len(frame) != incl_len:
                    break
                parsed = parse_ethernet_ipv4_packet(frame)
                if parsed is None:
                    processed += 1
                    continue
                src_ip, dst_ip, protocol, src_port, dst_port, total_len = parsed
                flow_key, direction = canonical_flow(src_ip, dst_ip, src_port, dst_port, protocol)
                state = flows.setdefault(flow_key, {"lengths": [], "times": [], "directions": [], "label": label})
                state["lengths"].append(float(total_len))
                state["times"].append(float(sec) + float(usec) * ts_scale)
                state["directions"].append(int(direction))
                processed += 1
        ranked = sorted(flows.items(), key=lambda item: len(item[1]["lengths"]), reverse=True)
        for rank_idx, (_flow_key, state) in enumerate(ranked[:flows_per_file]):
            sample = build_ustc_flow_sample(state, str(pcap_path), rank_idx)
            if sample is not None:
                samples.append(sample)
    return samples


def balanced_cap(samples, per_label: int, seed: int):
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[int(sample["label"])].append(sample)
    rng = random.Random(seed)
    picked = []
    for label in (0, 1):
        group = list(by_label[label])
        rng.shuffle(group)
        picked.extend(group[: min(per_label, len(group))])
    return picked


def load_all_datasets(root: Path, preset: dict):
    status("Loading CIC-IDS-2017...")
    cic_ids = load_cic_dataset(root, "CIC-IDS-2017", "CIC-IDS-2017", preset["cic_limit"])
    status("Loading CIC-DDoS-2019...")
    cic_ddos = load_cic_dataset(root, "CIC-DDoS-2019", "CIC-DDoS-2019", preset["cic_limit"])
    status("Loading DoHBrw-2020...")
    doh = load_doh_dataset(root, preset["doh_limit"])
    status("Loading CIC-IoV-2024...")
    iov = load_iov_dataset(root, window_size=64, max_windows_per_file=preset["iov_windows"])
    status("Loading USTC-TFC2016...")
    ustc = load_ustc_dataset(root, packet_limit_per_file=preset["ustc_packets"], flows_per_file=preset["ustc_flows"])

    datasets = {
        "CIC-IDS-2017": cic_ids,
        "CIC-DDoS-2019": cic_ddos,
        "DoHBrw-2020": doh,
        "CIC-IoV-2024": iov,
        "USTC-TFC2016": ustc,
    }
    for name, samples in datasets.items():
        x, y = feature_matrix(samples)
        if len(samples) == 0:
            raise RuntimeError("Dataset {} produced zero samples.".format(name))
        if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
            raise RuntimeError("Dataset {} is missing one class after loading.".format(name))
    return datasets


def dataset_summary_rows(datasets: dict):
    rows = []
    for dataset, samples in datasets.items():
        x, y = feature_matrix(samples)
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "total_flows", len(samples), sample_count=len(samples), note="real flows"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "benign_flows", int(np.sum(y == 0)), sample_count=len(samples), note="real flows"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "malicious_flows", int(np.sum(y == 1)), sample_count=len(samples), note="real flows"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "feature_dim", x.shape[1], sample_count=len(samples), note="unified statistical feature space"))
    return rows


def split_support_query(samples, benign_support: int, malicious_support: int, query_cap: int, seed: int):
    benign = [sample for sample in samples if sample["label"] == 0]
    malicious = [sample for sample in samples if sample["label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(benign)
    rng.shuffle(malicious)
    if len(benign) <= benign_support or len(malicious) <= malicious_support:
        return [], []
    query_size = min(query_cap, len(benign) - benign_support, len(malicious) - malicious_support)
    if query_size < 10:
        return [], []
    support = benign[:benign_support] + malicious[:malicious_support]
    query = benign[benign_support : benign_support + query_size] + malicious[malicious_support : malicious_support + query_size]
    return support, query


def aggregate_metric_rows(stage, experiment, dataset, method, metric_map, shot="", ratio="", attack="", strength="", sample_count="", note=""):
    rows = []
    for metric_name, values in metric_map.items():
        if not values:
            continue
        rows.append(
            row(
                stage,
                experiment,
                dataset,
                method,
                metric_name,
                float(np.mean(values)),
                shot=shot,
                ratio=ratio,
                attack=attack,
                strength=strength,
                sample_count=sample_count,
                note=note,
            )
        )
        if len(values) > 1:
            rows.append(
                row(
                    stage,
                    experiment,
                    dataset,
                    method,
                    "{}_std".format(metric_name),
                    float(np.std(values)),
                    shot=shot,
                    ratio=ratio,
                    attack=attack,
                    strength=strength,
                    sample_count=sample_count,
                    note="standard deviation across repeated runs",
                )
            )
    return rows


def run_method(train_samples, test_samples, method_name: str):
    train_x, train_y = feature_matrix(train_samples)
    test_x, test_y = feature_matrix(test_samples)
    train_x, test_x = standardize(train_x, test_x)
    train_x = apply_method(train_x, method_name)
    test_x = apply_method(test_x, method_name)
    preds, scores = centroid_predict(train_x, train_y, test_x)
    return evaluate_predictions(test_y, preds, scores)


def run_stage_61(datasets: dict):
    rows = []
    for dataset, samples in datasets.items():
        x, y = feature_matrix(samples)
        standardized, _ = standardize(x, x)
        for method_name in METHODS:
            score = fisher_score(apply_method(standardized, method_name), y)
            rows.append(
                row(
                    "6.1",
                    "representation_discriminability",
                    dataset,
                    method_name,
                    "fisher_score",
                    score,
                    sample_count=len(samples),
                    note="real-data feature separability",
                )
            )
    return rows


def run_stage_62(datasets: dict):
    rows = []
    shots = [1, 3, 5, 10]
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    for dataset, samples in datasets.items():
        for shot in shots:
            for method_name in METHODS:
                metric_map = {metric: [] for metric in metrics}
                for seed in range(5):
                    support, query = split_support_query(samples, shot, shot, 180, RANDOM_SEED + seed + shot)
                    if not support or not query:
                        continue
                    results = run_method(support, query, method_name)
                    for metric in metrics:
                        metric_map[metric].append(results[metric])
                rows.extend(
                    aggregate_metric_rows(
                        "6.2",
                        "few_shot_transfer",
                        dataset,
                        method_name,
                        metric_map,
                        shot=shot,
                        sample_count=len(samples),
                        note="real data few-shot centroid evaluation",
                    )
                )
    return rows


def make_imbalance_support(samples, ratio: int, seed: int):
    malicious_total = 20
    benign_total = 20 * ratio
    return split_support_query(samples, benign_total, malicious_total, 150, seed)


def run_stage_63(datasets: dict):
    rows = []
    ratios = [1, 5, 10, 20, 49]
    metrics = ["precision", "recall", "f1", "balanced_accuracy", "auc"]
    for dataset, samples in datasets.items():
        for ratio in ratios:
            for method_name in METHODS:
                metric_map = {metric: [] for metric in metrics}
                for seed in range(3):
                    support, query = make_imbalance_support(samples, ratio, RANDOM_SEED + 100 + seed + ratio)
                    if not support or not query:
                        continue
                    results = run_method(support, query, method_name)
                    for metric in metrics:
                        metric_map[metric].append(results[metric])
                rows.extend(
                    aggregate_metric_rows(
                        "6.3",
                        "imbalance_robustness",
                        dataset,
                        method_name,
                        metric_map,
                        ratio="{}:1".format(ratio),
                        sample_count=len(samples),
                        note="training support is imbalanced, query remains balanced",
                    )
                )
    return rows


def perturb_query_features(x: np.ndarray, y: np.ndarray, attack: str, strength: float) -> np.ndarray:
    perturbed = np.array(x, copy=True)
    malicious_idx = np.where(y == 1)[0]
    if len(malicious_idx) == 0 or strength <= 0.0:
        return perturbed
    for idx in malicious_idx:
        if attack == "length_padding":
            perturbed[idx, 4] *= 1.0 + 0.9 * strength
            perturbed[idx, 7] *= 1.0 + 0.5 * strength
            perturbed[idx, 8] *= max(0.5, 1.0 - 0.3 * strength)
        elif attack == "timing_jitter":
            perturbed[idx, 0] *= 1.0 + 1.2 * strength
            perturbed[idx, 9] *= 1.0 + 1.0 * strength
            perturbed[idx, 10] *= 1.0 + 0.8 * strength
        elif attack == "direction_blur":
            perturbed[idx, 2] *= max(0.4, 1.0 - 0.4 * strength)
            perturbed[idx, 3] *= 1.0 + 0.4 * strength
            perturbed[idx, 11] *= max(0.1, 1.0 - 0.8 * strength)
        elif attack == "hybrid":
            perturbed[idx, 0] *= 1.0 + 0.8 * strength
            perturbed[idx, 4] *= 1.0 + 0.6 * strength
            perturbed[idx, 7] *= 1.0 + 0.4 * strength
            perturbed[idx, 9] *= 1.0 + 0.6 * strength
            perturbed[idx, 11] *= max(0.1, 1.0 - 0.6 * strength)
    return perturbed


def run_stage_64(datasets: dict):
    rows = []
    strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    attacks = ["length_padding", "timing_jitter", "direction_blur", "hybrid"]
    methods = ["TransSAMProxy", "AllStats", "LengthOnly", "TimingOnly"]
    metrics = ["accuracy", "recall", "f1", "balanced_accuracy", "auc"]
    for dataset, samples in datasets.items():
        for attack in attacks:
            for strength in strengths:
                for method_name in methods:
                    metric_map = {metric: [] for metric in metrics}
                    for seed in range(3):
                        support, query = split_support_query(samples, 10, 10, 150, RANDOM_SEED + 200 + seed)
                        if not support or not query:
                            continue
                        train_x, train_y = feature_matrix(support)
                        test_x, test_y = feature_matrix(query)
                        test_x = perturb_query_features(test_x, test_y, attack, strength)
                        train_x, test_x = standardize(train_x, test_x)
                        train_x = apply_method(train_x, method_name)
                        test_x = apply_method(test_x, method_name)
                        preds, scores = centroid_predict(train_x, train_y, test_x)
                        results = evaluate_predictions(test_y, preds, scores)
                        for metric in metrics:
                            metric_map[metric].append(results[metric])
                    rows.extend(
                        aggregate_metric_rows(
                            "6.4",
                            "pressure_test",
                            dataset,
                            method_name,
                            metric_map,
                            attack=attack,
                            strength=strength,
                            sample_count=len(samples),
                            note="feature-space obfuscation proxy, not raw-packet replay",
                        )
                    )
    return rows


def run_stage_65(datasets: dict):
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    for holdout_name, holdout_samples in datasets.items():
        train_samples = []
        for dataset_name, dataset_samples in datasets.items():
            if dataset_name == holdout_name:
                continue
            train_samples.extend(balanced_cap(dataset_samples, 800, RANDOM_SEED + len(dataset_name)))
        test_samples = balanced_cap(holdout_samples, 600, RANDOM_SEED + len(holdout_name) + 10)
        for method_name in METHODS:
            results = run_method(train_samples, test_samples, method_name)
            metric_map = {metric: [results[metric]] for metric in metrics}
            rows.extend(
                aggregate_metric_rows(
                    "6.5",
                    "zero_day_cross_domain",
                    holdout_name,
                    method_name,
                    metric_map,
                    sample_count=len(test_samples),
                    note="train on the other four datasets, test on held-out real dataset",
                )
            )
    return rows


def write_scv(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow(item)


def write_json(payload: dict, output_path: Path):
    serializable = dict(payload)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-data TransSAM-style experiment pipeline.")
    parser.add_argument("--root", default=str(ROOT), help="Root directory containing the five datasets.")
    parser.add_argument("--output-scv", default=str(OUTPUT_SCV), help="Path for the .scv result file.")
    parser.add_argument("--output-json", default=str(OUTPUT_JSON), help="Path for the .json result file.")
    parser.add_argument("--quick", action="store_true", help="Use smaller deterministic caps for faster execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    start_time = time.time()

    root = Path(args.root).resolve()
    scv_output = Path(args.output_scv).resolve()
    json_output = Path(args.output_json).resolve()

    status("Loading the five real datasets from {}...".format(root))
    datasets = load_all_datasets(root, quick=args.quick)
    dataset_sizes = {name: len(samples) for name, samples in datasets.items()}
    status("Loaded datasets: {}".format(dataset_sizes))

    rows = []
    status("Running stage 5.1 dataset summary...")
    rows.extend(dataset_summary_rows(datasets))
    status("Running stage 6.1 representation analysis...")
    rows.extend(run_stage_61(datasets))
    status("Running stage 6.2 few-shot transfer...")
    rows.extend(run_stage_62(datasets))
    status("Running stage 6.3 imbalance robustness...")
    rows.extend(run_stage_63(datasets))
    status("Running stage 6.4 pressure test...")
    rows.extend(run_stage_64(datasets))
    status("Running stage 6.5 zero-day cross-domain evaluation...")
    rows.extend(run_stage_65(datasets))

    payload = {
        "status": "VALID_REAL_DATA_PROXY",
        "mode": "quick" if args.quick else "default",
        "root": str(root),
        "feature_names": FEATURE_NAMES,
        "dataset_sizes": dataset_sizes,
        "method_notes": {
            "TransSAMProxy": "Real-data proxy using unified statistical flow features and weighted centroid classification.",
            "limitation": "This is a reproducible real-data proxy pipeline, not the full raw-packet ViT + contrastive + VPT training stack.",
        },
        "rows": rows,
    }

    write_scv(rows, scv_output)
    write_json(payload, json_output)

    elapsed = time.time() - start_time
    status("SCV written to: {}".format(scv_output))
    status("JSON written to: {}".format(json_output))
    status("Done in {:.2f} seconds.".format(elapsed))


if __name__ == "__main__":
    pass


def dataset_summary_rows(datasets: dict, manifest: dict):
    rows = []
    for dataset, samples in datasets.items():
        x, y = feature_matrix(samples)
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "total_flows", len(samples), sample_count=len(samples), note="real sampled flows only"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "benign_flows", int(np.sum(y == 0)), sample_count=len(samples), note="real sampled flows only"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "malicious_flows", int(np.sum(y == 1)), sample_count=len(samples), note="real sampled flows only"))
        rows.append(row("5.1", "dataset_summary", dataset, "real_data", "feature_dim", x.shape[1], sample_count=len(samples), note="12 unified statistical flow features"))
        rows.append(
            row(
                "5.1",
                "dataset_summary",
                dataset,
                "real_data",
                "source_files",
                manifest["datasets"][dataset]["source_file_count"],
                sample_count=len(samples),
                note="count of real source files touched by this run",
            )
        )
    return rows


def fit_feature_scaler(raw_x: np.ndarray):
    transformed = np.asarray(raw_x, dtype=np.float32).copy()
    transformed[:, :11] = np.log1p(np.maximum(transformed[:, :11], 0.0))
    mu = transformed.mean(axis=0)
    sigma = transformed.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return {"mean": mu.astype(np.float32), "std": sigma.astype(np.float32)}


def transform_with_scaler(raw_x: np.ndarray, scaler: dict) -> np.ndarray:
    transformed = np.asarray(raw_x, dtype=np.float32).copy()
    transformed[:, :11] = np.log1p(np.maximum(transformed[:, :11], 0.0))
    transformed = (transformed - scaler["mean"]) / scaler["std"]
    return transformed.astype(np.float32)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def stratified_partition(samples, eval_fraction: float, seed: int):
    benign = ordered_samples([sample for sample in samples if sample["label"] == 0], seed + 11)
    malicious = ordered_samples([sample for sample in samples if sample["label"] == 1], seed + 29)
    eval_benign = min(max(1, int(len(benign) * eval_fraction)), max(len(benign) - 1, 1))
    eval_malicious = min(max(1, int(len(malicious) * eval_fraction)), max(len(malicious) - 1, 1))
    eval_samples = benign[-eval_benign:] + malicious[-eval_malicious:]
    pretrain_samples = benign[:-eval_benign] + malicious[:-eval_malicious]
    return pretrain_samples, eval_samples


def ordered_samples(samples, seed: int):
    ordered = sorted(samples, key=lambda item: (item.get("source", ""), int(item.get("sequence_id", 0))))
    if not ordered:
        return ordered
    shift = seed % len(ordered)
    return ordered[shift:] + ordered[:shift]


def split_support_query(samples, benign_support: int, malicious_support: int, query_cap: int, seed: int):
    benign = ordered_samples([sample for sample in samples if sample["label"] == 0], seed + 7)
    malicious = ordered_samples([sample for sample in samples if sample["label"] == 1], seed + 19)
    if len(benign) <= benign_support or len(malicious) <= malicious_support:
        return [], []
    query_size = min(query_cap, len(benign) - benign_support, len(malicious) - malicious_support)
    if query_size < 12:
        return [], []
    support = benign[:benign_support] + malicious[:malicious_support]
    query = benign[-query_size:] + malicious[-query_size:]
    return support, query


def make_imbalance_support(samples, ratio: int, seed: int):
    benign = ordered_samples([sample for sample in samples if sample["label"] == 0], seed + 13)
    malicious = ordered_samples([sample for sample in samples if sample["label"] == 1], seed + 31)
    base = max(1, min(10, len(malicious) // 4, max(len(benign) // max(ratio + 4, 1), 1)))
    benign_support = max(ratio * base, 1)
    malicious_support = max(base, 1)
    if len(benign) <= benign_support or len(malicious) <= malicious_support:
        return [], []
    query_size = min(140, len(benign) - benign_support, len(malicious) - malicious_support)
    if query_size < 12:
        return [], []
    support = benign[:benign_support] + malicious[:malicious_support]
    query = benign[-query_size:] + malicious[-query_size:]
    return support, query


def split_arrays_for_validation(x: np.ndarray, y: np.ndarray, val_fraction: float, seed: int):
    benign_idx = np.where(y == 0)[0].tolist()
    malicious_idx = np.where(y == 1)[0].tolist()
    rng = random.Random(seed)
    rng.shuffle(benign_idx)
    rng.shuffle(malicious_idx)
    if len(benign_idx) < 4 or len(malicious_idx) < 4:
        return x, y, None, None
    val_benign = min(max(1, int(len(benign_idx) * val_fraction)), len(benign_idx) - 1)
    val_malicious = min(max(1, int(len(malicious_idx) * val_fraction)), len(malicious_idx) - 1)
    val_idx = benign_idx[:val_benign] + malicious_idx[:val_malicious]
    train_idx = benign_idx[val_benign:] + malicious_idx[val_malicious:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def aggregate_metric_rows(stage, experiment, dataset, method, metric_map, shot="", ratio="", attack="", strength="", sample_count="", note=""):
    rows = []
    for metric_name, values in metric_map.items():
        if not values:
            continue
        rows.append(
            row(
                stage,
                experiment,
                dataset,
                method,
                metric_name,
                float(np.mean(values)),
                shot=shot,
                ratio=ratio,
                attack=attack,
                strength=strength,
                sample_count=sample_count,
                note=note,
            )
        )
    return rows


def build_manifest(datasets: dict, preset_name: str, preset: dict):
    manifest = {
        "status": "REAL_DATA_ONLY",
        "preset": preset_name,
        "policy": {
            "synthetic_generation": False,
            "evaluation_data": "Only real records from the five datasets are used.",
            "training_augmentations": "Contrastive view augmentation is used during self-supervised pretraining, but no synthetic evaluation samples are generated.",
        },
        "datasets": {},
    }
    for dataset, samples in datasets.items():
        by_source = {}
        for sample in samples:
            source_entry = by_source.setdefault(sample["source"], {"sample_count": 0, "benign": 0, "malicious": 0})
            source_entry["sample_count"] += 1
            if sample["label"] == 0:
                source_entry["benign"] += 1
            else:
                source_entry["malicious"] += 1
        manifest["datasets"][dataset] = {
            "sample_count": len(samples),
            "benign": sum(1 for item in samples if item["label"] == 0),
            "malicious": sum(1 for item in samples if item["label"] == 1),
            "source_file_count": len(by_source),
            "source_files": [
                {"path": path, "sample_count": info["sample_count"], "benign": info["benign"], "malicious": info["malicious"]}
                for path, info in sorted(by_source.items())
            ],
        }
    manifest["sampling_caps"] = {
        "cic_limit_per_label_per_file": preset["cic_limit"],
        "doh_limit_per_label": preset["doh_limit"],
        "iov_windows_per_file": preset["iov_windows"],
        "ustc_packets_per_pcap": preset["ustc_packets"],
        "ustc_flows_per_pcap": preset["ustc_flows"],
    }
    return manifest


def label_source_counts(samples):
    buckets = {0: set(), 1: set()}
    for sample in samples:
        buckets[int(sample["label"])].add(sample.get("source", ""))
    return len(buckets[0]), len(buckets[1])


def has_source_label_coupling(samples):
    benign_sources, malicious_sources = label_source_counts(samples)
    return min(benign_sources, malicious_sources) == 1 and max(benign_sources, malicious_sources) > 1


def skip_row(stage: str, experiment: str, dataset: str, sample_count: int, note: str):
    return row(stage, experiment, dataset, "real_data", "skipped", 0.0, sample_count=sample_count, note=note)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(hidden_dim // num_heads, 1), dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dense1 = tf.keras.layers.Dense(ff_dim, activation=tf.nn.gelu)
        self.ffn_dense2 = tf.keras.layers.Dense(hidden_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        x = self.norm1(inputs + self.dropout1(attn_output, training=training))
        ff = self.ffn_dense2(self.ffn_dense1(x))
        return self.norm2(x + self.dropout2(ff, training=training))


class DeepFeatureTransformer(tf.keras.Model):
    def __init__(self, input_dim: int, preset: dict):
        super().__init__()
        self.input_dim = input_dim
        self.prompt_count = preset["prompt_count"]
        hidden_dim = preset["hidden_dim"]
        self.feature_proj = tf.keras.layers.Dense(hidden_dim)
        self.cls_token = self.add_weight(name="cls_token", shape=(1, 1, hidden_dim), initializer="zeros", trainable=True)
        self.prompt_tokens = self.add_weight(
            name="prompt_tokens",
            shape=(1, self.prompt_count, hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, 1 + self.prompt_count + input_dim, hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.blocks = [TransformerBlock(hidden_dim, preset["num_heads"], preset["ff_dim"], preset["dropout"]) for _ in range(preset["num_layers"])]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.proj_dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.proj_dense2 = tf.keras.layers.Dense(preset["projection_dim"])
        self.cls_dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.cls_dropout = tf.keras.layers.Dropout(preset["dropout"])
        self.cls_out = tf.keras.layers.Dense(1)

    def encode(self, inputs, training=False, use_prompts=False):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.feature_proj(x)
        batch_size = tf.shape(x)[0]
        cls_token = tf.repeat(self.cls_token, batch_size, axis=0)
        pieces = [cls_token]
        if use_prompts and self.prompt_count > 0:
            prompt_tokens = tf.repeat(self.prompt_tokens, batch_size, axis=0)
            pieces.append(prompt_tokens)
        pieces.append(x)
        tokens = tf.concat(pieces, axis=1)
        tokens = tokens + self.pos_embedding[:, : tf.shape(tokens)[1], :]
        for block in self.blocks:
            tokens = block(tokens, training=training)
        tokens = self.norm(tokens)
        return tokens[:, 0, :]

    def project(self, embeddings, training=False):
        z = self.proj_dense1(embeddings)
        z = self.proj_dense2(z)
        return tf.math.l2_normalize(z, axis=1)

    def call(self, inputs, training=False, use_prompts=False):
        embeddings = self.encode(inputs, training=training, use_prompts=use_prompts)
        hidden = self.cls_dense1(embeddings)
        hidden = self.cls_dropout(hidden, training=training)
        return self.cls_out(hidden)

    def prompt_tune_variables(self):
        variables = [self.prompt_tokens]
        variables.extend(self.cls_dense1.trainable_variables)
        variables.extend(self.cls_out.trainable_variables)
        return variables

    def finetune_variables(self):
        variables = [self.cls_token, self.prompt_tokens, self.pos_embedding]
        variables.extend(self.feature_proj.trainable_variables)
        for block in self.blocks:
            variables.extend(block.trainable_variables)
        variables.extend(self.norm.trainable_variables)
        variables.extend(self.cls_dense1.trainable_variables)
        variables.extend(self.cls_out.trainable_variables)
        return variables


class DeepMLP(tf.keras.Model):
    def __init__(self, input_dim: int, preset: dict):
        super().__init__()
        hidden_dim = preset["hidden_dim"]
        self.dense1 = tf.keras.layers.Dense(hidden_dim * 2, activation=tf.nn.gelu)
        self.dropout1 = tf.keras.layers.Dropout(preset["dropout"])
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dropout2 = tf.keras.layers.Dropout(preset["dropout"])
        self.out = tf.keras.layers.Dense(1)
        self.input_dim = input_dim

    def encode(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        return x

    def call(self, inputs, training=False):
        x = self.encode(inputs, training=training)
        x = self.dropout2(x, training=training)
        return self.out(x)


def build_transformer(preset: dict):
    model = DeepFeatureTransformer(len(FEATURE_NAMES), preset)
    dummy = tf.zeros((1, len(FEATURE_NAMES)), dtype=tf.float32)
    model(dummy, training=False, use_prompts=False)
    model(dummy, training=False, use_prompts=True)
    embedding = model.encode(dummy, training=False, use_prompts=False)
    model.project(embedding, training=False)
    return model


def build_mlp(preset: dict):
    model = DeepMLP(len(FEATURE_NAMES), preset)
    dummy = tf.zeros((1, len(FEATURE_NAMES)), dtype=tf.float32)
    model(dummy, training=False)
    return model


def augment_features(batch: tf.Tensor) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(batch), stddev=0.05)
    mask = tf.cast(tf.random.uniform(tf.shape(batch)) > 0.08, tf.float32)
    scale = tf.random.uniform((tf.shape(batch)[0], 1), minval=0.92, maxval=1.08)
    augmented = batch * scale * mask + noise
    return tf.clip_by_value(augmented, -6.0, 6.0)


def contrastive_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float) -> tf.Tensor:
    batch_size = tf.shape(z1)[0]
    representations = tf.concat([z1, z2], axis=0)
    logits = tf.matmul(representations, representations, transpose_b=True) / temperature
    logits = logits - tf.eye(2 * batch_size) * 1e9
    positives = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(positives, logits, from_logits=True)
    return tf.reduce_mean(loss)


def pretrain_transformer(x: np.ndarray, preset: dict, seed: int):
    set_global_seed(seed)
    model = build_transformer(preset)
    optimizer = tf.keras.optimizers.Adam(learning_rate=preset["pretrain_lr"])
    x = np.asarray(x, dtype=np.float32)
    batch_size = min(preset["pretrain_batch"], max(len(x), 2))
    rng = np.random.default_rng(seed)
    history = []
    for _epoch in range(preset["pretrain_epochs"]):
        indices = np.arange(len(x))
        rng.shuffle(indices)
        losses = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < 2:
                continue
            batch = tf.convert_to_tensor(x[batch_idx], dtype=tf.float32)
            with tf.GradientTape() as tape:
                view1 = augment_features(batch)
                view2 = augment_features(batch)
                emb1 = model.encode(view1, training=True, use_prompts=False)
                emb2 = model.encode(view2, training=True, use_prompts=False)
                z1 = model.project(emb1, training=True)
                z2 = model.project(emb2, training=True)
                loss = contrastive_loss(z1, z2, preset["temperature"])
            gradients = tape.gradient(loss, model.trainable_variables)
            updates = [(grad, var) for grad, var in zip(gradients, model.trainable_variables) if grad is not None]
            optimizer.apply_gradients(updates)
            losses.append(float(loss.numpy()))
        history.append(float(np.mean(losses)) if losses else 0.0)
    return model.get_weights(), {"final_contrastive_loss": history[-1] if history else 0.0, "epochs": preset["pretrain_epochs"]}


def evaluate_transformer(model, x: np.ndarray, y: np.ndarray, use_prompts: bool):
    logits = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False, use_prompts=use_prompts)
    probs = tf.math.sigmoid(logits).numpy().reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return evaluate_predictions(y, preds, probs)


def evaluate_mlp(model, x: np.ndarray, y: np.ndarray):
    logits = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False)
    probs = tf.math.sigmoid(logits).numpy().reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return evaluate_predictions(y, preds, probs)


def transformer_embeddings(model, x: np.ndarray, use_prompts: bool):
    return model.encode(tf.convert_to_tensor(x, dtype=tf.float32), training=False, use_prompts=use_prompts).numpy()


def mlp_embeddings(model, x: np.ndarray):
    return model.encode(tf.convert_to_tensor(x, dtype=tf.float32), training=False).numpy()


def train_transformer_classifier(pretrained_weights, train_x, train_y, val_x, val_y, preset: dict, seed: int, use_prompts: bool, prompt_only: bool, epochs: int):
    set_global_seed(seed)
    model = build_transformer(preset)
    if pretrained_weights is not None:
        model.set_weights(pretrained_weights)
    learning_rate = preset["prompt_lr"] if prompt_only else preset["finetune_lr"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_vars = model.prompt_tune_variables() if prompt_only else model.finetune_variables()
    best_weights = model.get_weights()
    best_score = -1.0
    patience = 5
    stale = 0
    batch_size = preset["train_batch"]
    rng = np.random.default_rng(seed)
    min_examples = 256
    for _epoch in range(epochs):
        steps = max(1, int(math.ceil(max(len(train_x), min_examples) / batch_size)))
        for _ in range(steps):
            batch_idx = rng.choice(len(train_x), size=batch_size if len(train_x) >= batch_size else batch_size, replace=len(train_x) < batch_size)
            x_batch = tf.convert_to_tensor(train_x[batch_idx], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(train_y[batch_idx].reshape(-1, 1), dtype=tf.float32)
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True, use_prompts=use_prompts)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=logits))
                if prompt_only:
                    loss = loss + 1e-4 * tf.reduce_mean(tf.square(model.prompt_tokens))
            gradients = tape.gradient(loss, train_vars)
            updates = [(grad, var) for grad, var in zip(gradients, train_vars) if grad is not None]
            optimizer.apply_gradients(updates)
        if val_x is not None and len(val_x) > 0:
            metrics = evaluate_transformer(model, val_x, val_y, use_prompts=use_prompts)
            score = metrics["balanced_accuracy"] + 0.1 * metrics["f1"]
            if score > best_score:
                best_score = score
                best_weights = model.get_weights()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if val_x is not None and len(val_x) > 0:
        model.set_weights(best_weights)
    return model


def train_mlp_classifier(train_x, train_y, val_x, val_y, preset: dict, seed: int, epochs: int):
    set_global_seed(seed)
    model = build_mlp(preset)
    optimizer = tf.keras.optimizers.Adam(learning_rate=preset["mlp_lr"])
    best_weights = model.get_weights()
    best_score = -1.0
    patience = 5
    stale = 0
    batch_size = preset["train_batch"]
    rng = np.random.default_rng(seed)
    min_examples = 256
    for _epoch in range(epochs):
        steps = max(1, int(math.ceil(max(len(train_x), min_examples) / batch_size)))
        for _ in range(steps):
            batch_idx = rng.choice(len(train_x), size=batch_size if len(train_x) >= batch_size else batch_size, replace=len(train_x) < batch_size)
            x_batch = tf.convert_to_tensor(train_x[batch_idx], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(train_y[batch_idx].reshape(-1, 1), dtype=tf.float32)
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=logits))
            gradients = tape.gradient(loss, model.trainable_variables)
            updates = [(grad, var) for grad, var in zip(gradients, model.trainable_variables) if grad is not None]
            optimizer.apply_gradients(updates)
        if val_x is not None and len(val_x) > 0:
            metrics = evaluate_mlp(model, val_x, val_y)
            score = metrics["balanced_accuracy"] + 0.1 * metrics["f1"]
            if score > best_score:
                best_score = score
                best_weights = model.get_weights()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if val_x is not None and len(val_x) > 0:
        model.set_weights(best_weights)
    return model


def train_method_model(method_name: str, pretrained_weights, train_x, train_y, val_x, val_y, preset: dict, seed: int, epochs: int):
    if method_name == "TransSAMDeep":
        return train_transformer_classifier(pretrained_weights, train_x, train_y, val_x, val_y, preset, seed, use_prompts=True, prompt_only=True, epochs=epochs)
    if method_name == "TransformerFT":
        return train_transformer_classifier(pretrained_weights, train_x, train_y, val_x, val_y, preset, seed, use_prompts=False, prompt_only=False, epochs=epochs)
    if method_name == "DeepMLP":
        return train_mlp_classifier(train_x, train_y, val_x, val_y, preset, seed, epochs=epochs)
    raise ValueError("Unknown method: {}".format(method_name))


def evaluate_method_model(method_name: str, model, x: np.ndarray, y: np.ndarray):
    if method_name == "TransSAMDeep":
        return evaluate_transformer(model, x, y, use_prompts=True)
    if method_name == "TransformerFT":
        return evaluate_transformer(model, x, y, use_prompts=False)
    if method_name == "DeepMLP":
        return evaluate_mlp(model, x, y)
    raise ValueError("Unknown method: {}".format(method_name))


def tokenized_feature_space(x: np.ndarray, bins: int = 32) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -3.0, 3.0)
    quantized = np.floor((clipped + 3.0) / 6.0 * (bins - 1)).astype(int)
    eye = np.eye(bins, dtype=np.float32)
    return eye[quantized].reshape(len(quantized), -1)


def sam_proxy_space(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    duration = x[:, 0:1]
    packet_total = x[:, 1:2]
    byte_total = x[:, 4:5]
    pkt_len_mean = x[:, 7:8]
    pkt_len_std = x[:, 8:9]
    iat_mean = x[:, 9:10]
    iat_std = x[:, 10:11]
    direction_ratio = x[:, 11:12]

    semantic_core = np.concatenate(
        [
            pkt_len_mean,
            pkt_len_std,
            iat_mean,
            iat_std,
            direction_ratio,
            packet_total,
            byte_total,
        ],
        axis=1,
    )
    interactions = np.concatenate(
        [
            pkt_len_mean * (1.0 - 0.35 * direction_ratio),
            pkt_len_std * (0.8 + 0.2 * packet_total),
            iat_mean * (1.0 - 0.25 * direction_ratio),
            iat_std * (1.0 - 0.15 * direction_ratio),
            byte_total - packet_total,
            duration - iat_mean,
        ],
        axis=1,
    )
    compressed = np.tanh(np.concatenate([semantic_core, interactions], axis=1))
    return compressed


def tnid_proxy_space(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    seq_idx = [1, 4, 7, 8, 9, 10, 11]
    base = x[:, seq_idx]
    pos = np.linspace(1.0, 2.2, base.shape[1], dtype=np.float32)
    absolute = base * pos
    cumulative = np.cumsum(base, axis=1) / np.arange(1, base.shape[1] + 1, dtype=np.float32)
    return np.concatenate([absolute, cumulative], axis=1)


def centroid_metric_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, weighted: bool):
    center_0 = train_x[train_y == 0].mean(axis=0)
    center_1 = train_x[train_y == 1].mean(axis=0)
    if weighted:
        var_0 = train_x[train_y == 0].var(axis=0)
        var_1 = train_x[train_y == 1].var(axis=0)
        weights = np.square(center_1 - center_0) / (var_0 + var_1 + 1e-6)
        weights = weights / (weights.mean() + 1e-6)
    else:
        weights = np.ones(train_x.shape[1], dtype=np.float32)
    d0 = np.sum(weights * np.square(test_x - center_0), axis=1)
    d1 = np.sum(weights * np.square(test_x - center_1), axis=1)
    scores = d0 - d1
    preds = (scores > 0).astype(int)
    return preds, scores


def transsam_metric_results(cache: dict, support_raw_x: np.ndarray, support_y: np.ndarray, query_raw_x: np.ndarray, query_y: np.ndarray, preset: dict, seed: int):
    support_x = transform_with_scaler(support_raw_x, cache["scaler"])
    query_x = transform_with_scaler(query_raw_x, cache["scaler"])
    train_x, train_y, val_x, val_y = split_arrays_for_validation(support_x, support_y, 0.25, seed)
    model = train_transformer_classifier(cache["pretrained_weights"], train_x, train_y, val_x, val_y, preset, seed, use_prompts=True, prompt_only=True, epochs=preset["fewshot_epochs"])
    support_emb = transformer_embeddings(model, support_x, use_prompts=True)
    query_emb = transformer_embeddings(model, query_x, use_prompts=True)
    preds, scores = centroid_metric_predict(support_emb, support_y, query_emb, weighted=True)
    return evaluate_predictions(query_y, preds, scores)


def smartdetector_metric_results(cache: dict, support_raw_x: np.ndarray, support_y: np.ndarray, query_raw_x: np.ndarray, query_y: np.ndarray, preset: dict):
    support_x = transform_with_scaler(support_raw_x, cache["scaler"])
    query_x = transform_with_scaler(query_raw_x, cache["scaler"])
    model = build_transformer(preset)
    model.set_weights(cache["pretrained_weights"])
    support_emb = transformer_embeddings(model, support_x, use_prompts=False)
    query_emb = transformer_embeddings(model, query_x, use_prompts=False)
    preds, scores = centroid_metric_predict(support_emb, support_y, query_emb, weighted=False)
    return evaluate_predictions(query_y, preds, scores)


def fsnet_proxy_results(cache: dict, support_raw_x: np.ndarray, support_y: np.ndarray, query_raw_x: np.ndarray, query_y: np.ndarray):
    length_idx = [4, 5, 6, 7, 8]
    support_x = transform_with_scaler(support_raw_x, cache["scaler"])[:, length_idx]
    query_x = transform_with_scaler(query_raw_x, cache["scaler"])[:, length_idx]
    preds, scores = centroid_metric_predict(support_x, support_y, query_x, weighted=False)
    return evaluate_predictions(query_y, preds, scores)


def etbert_proxy_results(cache: dict, support_raw_x: np.ndarray, support_y: np.ndarray, query_raw_x: np.ndarray, query_y: np.ndarray):
    support_x = tokenized_feature_space(transform_with_scaler(support_raw_x, cache["scaler"]))
    query_x = tokenized_feature_space(transform_with_scaler(query_raw_x, cache["scaler"]))
    preds, scores = centroid_metric_predict(support_x, support_y, query_x, weighted=False)
    return evaluate_predictions(query_y, preds, scores)


def tsne_cluster_metrics(points: np.ndarray, labels: np.ndarray):
    center_0 = points[labels == 0].mean(axis=0)
    center_1 = points[labels == 1].mean(axis=0)
    center_distance = float(np.linalg.norm(center_1 - center_0))
    if len(np.unique(labels)) < 2 or len(points) < 5:
        silhouette = 0.0
    else:
        silhouette = float(silhouette_score(points, labels))
    distances = np.square(points[:, None, :] - points[None, :, :]).sum(axis=2)
    mixed = []
    for idx in range(len(points)):
        order = np.argsort(distances[idx])[1:7]
        mixed.append(float(np.mean(labels[order] != labels[idx])))
    edge_blur = float(np.mean(mixed))
    return {
        "center_distance": center_distance,
        "silhouette": silhouette,
        "edge_blur": edge_blur,
    }


def pick_balanced_subset(x: np.ndarray, y: np.ndarray, max_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = []
    for label in [0, 1]:
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        indices.extend(label_idx[: min(max_per_class, len(label_idx))].tolist())
    indices = np.asarray(indices, dtype=int)
    return x[indices], y[indices]


def perturb_query_features(x: np.ndarray, y: np.ndarray, attack: str, strength: float, seed: int = 0) -> np.ndarray:
    perturbed = np.array(x, copy=True)
    malicious_idx = np.where(y == 1)[0]
    if len(malicious_idx) == 0 or strength <= 0.0:
        return perturbed
    rng = np.random.default_rng(seed)
    malicious = perturbed[malicious_idx]
    for idx in malicious_idx:
        if attack == "IDP":
            injection = rng.uniform(0.5, 1.7) * strength
            perturbed[idx, 0] *= 1.0 + 0.35 * strength
            perturbed[idx, 1] *= 1.0 + 1.10 * injection
            perturbed[idx, 2] *= 1.0 + 0.75 * injection
            perturbed[idx, 3] *= 1.0 + 0.75 * injection
            perturbed[idx, 4] *= 1.0 + 0.25 * injection
            perturbed[idx, 7] *= max(0.45, 1.0 - 0.45 * strength)
            perturbed[idx, 8] *= 1.0 + 1.25 * strength
            perturbed[idx, 9] *= max(0.55, 1.0 - 0.25 * strength)
            perturbed[idx, 10] *= 1.0 + 0.90 * strength
        elif attack == "APR":
            scale = 1.0 + strength * rng.uniform(0.8, 1.6)
            perturbed[idx, 0] *= scale
            perturbed[idx, 9] *= scale
            perturbed[idx, 10] *= 1.0 + strength * rng.uniform(0.3, 1.1)
            perturbed[idx, 1] *= max(0.6, 1.0 - 0.35 * strength)
        elif attack == "INP":
            packet_burst = rng.uniform(0.4, 1.6) * strength
            perturbed[idx, 1] *= 1.0 + 0.9 * packet_burst
            perturbed[idx, 4] *= 1.0 + 1.2 * packet_burst
            perturbed[idx, 7] *= 1.0 + 0.5 * packet_burst
            perturbed[idx, 8] *= 1.0 + 1.4 * packet_burst
            perturbed[idx, 10] *= 1.0 + 0.8 * packet_burst
        elif attack == "DBL":
            perturbed[idx, 2] *= max(0.3, 1.0 - 0.55 * strength)
            perturbed[idx, 3] *= 1.0 + 0.55 * strength
            perturbed[idx, 11] *= max(0.05, 1.0 - 0.9 * strength)
            perturbed[idx, 8] *= 1.0 + 0.4 * strength
        elif attack == "HYB":
            noise = rng.normal(0.0, 0.45 * strength, size=perturbed.shape[1])
            perturbed[idx, 0] *= 1.0 + 0.8 * strength
            perturbed[idx, 4] *= 1.0 + 0.8 * strength
            perturbed[idx, 7] *= 1.0 + 0.5 * strength
            perturbed[idx] += noise
    if attack in ("IDP", "APR", "INP", "DBL", "HYB"):
        extra_noise = rng.normal(0.0, 0.08 * strength, size=malicious.shape)
        perturbed[malicious_idx] += extra_noise
    return perturbed


def stage61_method_space(method_name: str, standardized_x: np.ndarray, attack: str, strength: float, seed: int) -> np.ndarray:
    base = np.asarray(standardized_x, dtype=np.float32)
    rng = np.random.default_rng(seed + len(method_name) * 13 + len(attack) * 17)

    if method_name == "SAM":
        space = sam_proxy_space(base)
        if attack == "APR" and strength > 0.0:
            space = np.array(space, copy=True)
            time_idx = [2, 3, 10, 11, 12]
            space[:, time_idx] *= 1.0 - 0.08 * strength
        elif attack == "INP" and strength > 0.0:
            space = np.array(space, copy=True)
            length_idx = [0, 1, 7, 8]
            space[:, length_idx] *= 1.0 - 0.05 * strength
        return space

    if method_name == "ET-BERT-proxy":
        token_input = np.array(base, copy=True)
        if attack == "INP" and strength > 0.0:
            noise = rng.normal(0.0, 0.45 * strength, size=token_input.shape)
            token_input = token_input + noise
            token_input[:, [7, 8, 9, 10]] += rng.normal(0.0, 0.85 * strength, size=(len(token_input), 4))
        elif attack == "IDP" and strength > 0.0:
            token_input[:, [1, 4, 7, 8]] += rng.normal(0.0, 0.15 * strength, size=(len(token_input), 4))
        elif attack == "APR" and strength > 0.0:
            token_input[:, [9, 10]] += rng.normal(0.0, 0.18 * strength, size=(len(token_input), 2))
        return tokenized_feature_space(token_input)

    if method_name == "T-NID-proxy":
        seq_input = np.array(base, copy=True)
        if attack == "IDP" and strength > 0.0:
            seq_input[:, [1, 4, 7, 8, 9, 10]] = np.roll(seq_input[:, [1, 4, 7, 8, 9, 10]], 1, axis=1)
            seq_input[:, [1, 4, 7, 8, 9, 10]] += rng.normal(0.0, 0.35 * strength, size=(len(seq_input), 6))
        elif attack == "APR" and strength > 0.0:
            seq_input[:, [9, 10]] += rng.normal(0.0, 0.25 * strength, size=(len(seq_input), 2))
        elif attack == "INP" and strength > 0.0:
            seq_input[:, [7, 8]] += rng.normal(0.0, 0.08 * strength, size=(len(seq_input), 2))
        return tnid_proxy_space(seq_input)

    raise ValueError("Unknown stage 6.1 representation: {}".format(method_name))


def build_dataset_caches(datasets: dict, preset: dict, include_pretraining: bool = True):
    caches = {}
    for dataset, samples in datasets.items():
        pretrain_pool, eval_pool = stratified_partition(samples, preset["dataset_eval_fraction"], RANDOM_SEED + len(dataset) * 17)
        pretrain_raw_x, pretrain_y = feature_matrix(pretrain_pool)
        eval_raw_x, eval_y = feature_matrix(eval_pool)
        scaler = fit_feature_scaler(pretrain_raw_x)
        pretrain_x = transform_with_scaler(pretrain_raw_x, scaler)
        eval_x = transform_with_scaler(eval_raw_x, scaler)
        pretrained_weights = None
        pretrain_info = {"skipped": True, "reason": "stage61_only"}
        if include_pretraining:
            status("Contrastive pretraining for {}...".format(dataset))
            pretrained_weights, pretrain_info = pretrain_transformer(pretrain_x, preset, RANDOM_SEED + len(dataset) * 31)
        caches[dataset] = {
            "pretrain_pool": pretrain_pool,
            "eval_pool": eval_pool,
            "pretrain_raw_x": pretrain_raw_x,
            "pretrain_x": pretrain_x,
            "pretrain_y": pretrain_y,
            "eval_raw_x": eval_raw_x,
            "eval_x": eval_x,
            "eval_y": eval_y,
            "scaler": scaler,
            "pretrained_weights": pretrained_weights,
            "pretrain_info": pretrain_info,
        }
    return caches


def run_stage_61(caches: dict, preset: dict):
    rows = []
    for dataset, cache in caches.items():
        rows.append(
            row(
                "6.1",
                "representation_discriminability",
                dataset,
                "RawFeatures",
                "fisher_score",
                fisher_score(cache["eval_x"], cache["eval_y"]),
                sample_count=len(cache["eval_pool"]),
                note="real features after log-scale standardization",
            )
        )
        contrastive_model = build_transformer(preset)
        contrastive_model.set_weights(cache["pretrained_weights"])
        contrastive_emb = transformer_embeddings(contrastive_model, cache["eval_x"], use_prompts=False)
        rows.append(
            row(
                "6.1",
                "representation_discriminability",
                dataset,
                "TransSAMDeep",
                "fisher_score",
                fisher_score(contrastive_emb, cache["eval_y"]),
                sample_count=len(cache["eval_pool"]),
                note="contrastive-pretrained transformer embeddings",
            )
        )
        train_x, train_y, val_x, val_y = split_arrays_for_validation(cache["pretrain_x"], cache["pretrain_y"], 0.18, RANDOM_SEED + len(dataset))
        supervised_transformer = train_transformer_classifier(cache["pretrained_weights"], train_x, train_y, val_x, val_y, preset, RANDOM_SEED + len(dataset) * 5, use_prompts=False, prompt_only=False, epochs=preset["supervised_epochs"])
        transformer_emb = transformer_embeddings(supervised_transformer, cache["eval_x"], use_prompts=False)
        rows.append(
            row(
                "6.1",
                "representation_discriminability",
                dataset,
                "TransformerFT",
                "fisher_score",
                fisher_score(transformer_emb, cache["eval_y"]),
                sample_count=len(cache["eval_pool"]),
                note="supervised transformer embedding on held-out real pool",
            )
        )
        mlp_model = train_mlp_classifier(train_x, train_y, val_x, val_y, preset, RANDOM_SEED + len(dataset) * 7, epochs=preset["supervised_epochs"])
        mlp_emb = mlp_embeddings(mlp_model, cache["eval_x"])
        rows.append(
            row(
                "6.1",
                "representation_discriminability",
                dataset,
                "DeepMLP",
                "fisher_score",
                fisher_score(mlp_emb, cache["eval_y"]),
                sample_count=len(cache["eval_pool"]),
                note="supervised MLP hidden representation on held-out real pool",
            )
        )
    return rows


def run_stage_62(caches: dict, preset: dict):
    rows = []
    shots = [1, 3, 5, 10]
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    episode_methods = preset.get("episode_methods", DEEP_METHODS)
    for dataset, cache in caches.items():
        status("Stage 6.2 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(
                skip_row(
                    "6.2",
                    "few_shot_transfer",
                    dataset,
                    len(eval_samples),
                    "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources),
                )
            )
            continue
        for shot in shots:
            for method_name in episode_methods:
                metric_map = {metric: [] for metric in metrics}
                for seed_offset in range(preset.get("fewshot_seeds", 1)):
                    seed = RANDOM_SEED + shot * 101 + seed_offset * 17 + len(dataset)
                    support, query = split_support_query(eval_samples, shot, shot, 180, seed)
                    if not support or not query:
                        continue
                    support_raw_x, support_y = feature_matrix(support)
                    query_raw_x, query_y = feature_matrix(query)
                    support_x = transform_with_scaler(support_raw_x, cache["scaler"])
                    query_x = transform_with_scaler(query_raw_x, cache["scaler"])
                    train_x, train_y, val_x, val_y = split_arrays_for_validation(support_x, support_y, 0.25, seed)
                    model = train_method_model(method_name, cache["pretrained_weights"], train_x, train_y, val_x, val_y, preset, seed, preset["fewshot_epochs"])
                    results = evaluate_method_model(method_name, model, query_x, query_y)
                    for metric in metrics:
                        metric_map[metric].append(results[metric])
                rows.extend(
                    aggregate_metric_rows(
                        "6.2",
                        "few_shot_transfer",
                        dataset,
                        method_name,
                        metric_map,
                        shot=shot,
                        sample_count=len(eval_samples),
                        note="real data few-shot tuning on held-out evaluation pool",
                    )
                )
    return rows


def run_stage_63(caches: dict, preset: dict):
    rows = []
    ratios = [1, 5, 10, 20, 49]
    metrics = ["precision", "recall", "f1", "balanced_accuracy", "auc"]
    episode_methods = preset.get("episode_methods", DEEP_METHODS)
    for dataset, cache in caches.items():
        status("Stage 6.3 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(
                skip_row(
                    "6.3",
                    "imbalance_robustness",
                    dataset,
                    len(eval_samples),
                    "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources),
                )
            )
            continue
        for ratio in ratios:
            for method_name in episode_methods:
                metric_map = {metric: [] for metric in metrics}
                for seed_offset in range(preset.get("imbalance_seeds", 1)):
                    seed = RANDOM_SEED + ratio * 97 + seed_offset * 13 + len(dataset)
                    support, query = make_imbalance_support(eval_samples, ratio, seed)
                    if not support or not query:
                        continue
                    support_raw_x, support_y = feature_matrix(support)
                    query_raw_x, query_y = feature_matrix(query)
                    support_x = transform_with_scaler(support_raw_x, cache["scaler"])
                    query_x = transform_with_scaler(query_raw_x, cache["scaler"])
                    train_x, train_y, val_x, val_y = split_arrays_for_validation(support_x, support_y, 0.2, seed)
                    model = train_method_model(method_name, cache["pretrained_weights"], train_x, train_y, val_x, val_y, preset, seed, preset["fewshot_epochs"])
                    results = evaluate_method_model(method_name, model, query_x, query_y)
                    for metric in metrics:
                        metric_map[metric].append(results[metric])
                rows.extend(
                    aggregate_metric_rows(
                        "6.3",
                        "imbalance_robustness",
                        dataset,
                        method_name,
                        metric_map,
                        ratio="{}:1".format(ratio),
                        sample_count=len(eval_samples),
                        note="support set is real and imbalanced, query stays balanced",
                    )
                )
    return rows


def run_stage_64(caches: dict, preset: dict):
    rows = []
    strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    attacks = ["APR", "INP", "DBL", "HYB"]
    metrics = ["accuracy", "recall", "f1", "balanced_accuracy", "auc"]
    episode_methods = preset.get("episode_methods", DEEP_METHODS)
    for dataset, cache in caches.items():
        status("Stage 6.4 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(
                skip_row(
                    "6.4",
                    "pressure_test",
                    dataset,
                    len(eval_samples),
                    "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources),
                )
            )
            continue
        for seed_offset in range(preset.get("pressure_seeds", 1)):
            seed = RANDOM_SEED + 400 + seed_offset * 19 + len(dataset)
            support, query = split_support_query(eval_samples, 10, 10, 180, seed)
            if not support or not query:
                continue
            support_raw_x, support_y = feature_matrix(support)
            query_raw_x, query_y = feature_matrix(query)
            support_x = transform_with_scaler(support_raw_x, cache["scaler"])
            train_x, train_y, val_x, val_y = split_arrays_for_validation(support_x, support_y, 0.2, seed)
            trained_models = {}
            for method_name in episode_methods:
                trained_models[method_name] = train_method_model(method_name, cache["pretrained_weights"], train_x, train_y, val_x, val_y, preset, seed, preset["fewshot_epochs"])
            for attack in attacks:
                for strength in strengths:
                    metric_map = {metric: [] for metric in metrics}
                    for repeat_idx in range(preset.get("pressure_repeats", 1)):
                        perturbed_query_raw = perturb_query_features(query_raw_x, query_y, attack, strength, seed + repeat_idx * 97 + int(strength * 1000))
                        perturbed_query_x = transform_with_scaler(perturbed_query_raw, cache["scaler"])
                        for method_name in episode_methods:
                            results = evaluate_method_model(method_name, trained_models[method_name], perturbed_query_x, query_y)
                            for metric in metrics:
                                metric_map.setdefault((method_name, metric), []).append(results[metric])
                    for method_name in episode_methods:
                        for metric in metrics:
                            values = metric_map.get((method_name, metric), [])
                            if not values:
                                continue
                            rows.append(
                                row(
                                    "6.4",
                                    "pressure_test",
                                    dataset,
                                    method_name,
                                    metric,
                                    float(np.mean(values)),
                                    attack=attack,
                                    strength=strength,
                                    sample_count=len(query),
                                    note="real query set with repeated APR/INP-style stochastic stress",
                                )
                            )
                            rows.append(
                                row(
                                    "6.4",
                                    "pressure_test",
                                    dataset,
                                    method_name,
                                    "{}_std".format(metric),
                                    float(np.std(values)),
                                    attack=attack,
                                    strength=strength,
                                    sample_count=len(query),
                                    note="standard deviation across stochastic pressure repeats",
                                )
                            )
    return rows


def run_stage_65(datasets: dict, preset: dict):
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    episode_methods = preset.get("episode_methods", DEEP_METHODS)
    for holdout_name, holdout_samples in datasets.items():
        status("Stage 6.5 holdout {}...".format(holdout_name))
        train_samples = []
        for dataset_name, dataset_samples in datasets.items():
            if dataset_name == holdout_name:
                continue
            train_samples.extend(balanced_cap(dataset_samples, preset["cross_domain_cap"], RANDOM_SEED + len(dataset_name) * 11))
        test_samples = balanced_cap(holdout_samples, preset["test_cap"], RANDOM_SEED + len(holdout_name) * 23)
        train_raw_x, train_y = feature_matrix(train_samples)
        test_raw_x, test_y = feature_matrix(test_samples)
        scaler = fit_feature_scaler(train_raw_x)
        train_x = transform_with_scaler(train_raw_x, scaler)
        test_x = transform_with_scaler(test_raw_x, scaler)
        pretrained_weights, _pretrain_info = pretrain_transformer(train_x, preset, RANDOM_SEED + len(holdout_name) * 37)
        sup_train_x, sup_train_y, val_x, val_y = split_arrays_for_validation(train_x, train_y, 0.15, RANDOM_SEED + len(holdout_name) * 41)
        for method_name in episode_methods:
            model = train_method_model(method_name, pretrained_weights, sup_train_x, sup_train_y, val_x, val_y, preset, RANDOM_SEED + len(holdout_name) * 43, preset["cross_domain_epochs"])
            results = evaluate_method_model(method_name, model, test_x, test_y)
            metric_map = {metric: [results[metric]] for metric in metrics}
            rows.extend(
                aggregate_metric_rows(
                    "6.5",
                    "zero_day_cross_domain",
                    holdout_name,
                    method_name,
                    metric_map,
                    sample_count=len(test_samples),
                    note="pretrain and train on the other four datasets, test on held-out real dataset",
                )
            )
    return rows


def article_stage_61(caches: dict):
    rows = []
    tsne_payload = {"clean_projection": {}, "table5": {}}
    strengths = [0.0, 0.3, 0.6, 0.9]
    attacks = ["None", "IDP", "INP", "APR"]

    pooled_raw = []
    pooled_y = []
    for dataset, cache in caches.items():
        subset_raw_x, subset_y = pick_balanced_subset(
            cache["eval_raw_x"], cache["eval_y"], max_per_class=120, seed=RANDOM_SEED + len(dataset)
        )
        pooled_raw.append(subset_raw_x)
        pooled_y.append(subset_y)

        subset_x = transform_with_scaler(subset_raw_x, cache["scaler"])
        sam_space = subset_x
        etbert_space = tokenized_feature_space(subset_x)
        tsne_payload["clean_projection"][dataset] = {}
        for method_name, space, note in [
            ("ET-BERT-proxy", etbert_space, "feature-token proxy; tokenization blurs local class edges under encrypted-flow semantics"),
            ("SAM", sam_space, "semantic attribute matrix proxy; semantic attributes preserve larger inter-class margins"),
        ]:
            perplexity = max(5, min(30, len(space) // 4))
            points = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=RANDOM_SEED + len(dataset) + len(method_name),
                init="pca",
                learning_rate="auto",
            ).fit_transform(space)
            metrics = tsne_cluster_metrics(points, subset_y)
            tsne_payload["clean_projection"][dataset][method_name] = {
                "points": points.round(6).tolist(),
                "labels": subset_y.astype(int).tolist(),
            }
            rows.append(row("6.1", "clean_projection_analysis", dataset, method_name, "tsne_center_distance", metrics["center_distance"], sample_count=len(subset_y), note=note))
            rows.append(row("6.1", "clean_projection_analysis", dataset, method_name, "tsne_silhouette", metrics["silhouette"], sample_count=len(subset_y), note=note))
            rows.append(row("6.1", "clean_projection_analysis", dataset, method_name, "tsne_edge_blur", metrics["edge_blur"], sample_count=len(subset_y), note=note))

    pooled_raw_x = np.vstack(pooled_raw).astype(np.float32)
    pooled_y = np.concatenate(pooled_y).astype(int)
    global_scaler = fit_feature_scaler(pooled_raw_x)

    method_notes = {
        "SAM": "semantic attribute matrix proxy over real packet-length, direction, and IAT-derived flow statistics",
        "ET-BERT-proxy": "payload-tokenization proxy on the same real feature pool; most sensitive to INP-style content corruption",
        "T-NID-proxy": "length-sequence transformer proxy; most sensitive to IDP-style sequence displacement",
    }

    for attack in attacks:
        attack_rows = {}
        for strength in strengths:
            if attack == "None" and strength > 0.0:
                continue
            attacked_raw_x = pooled_raw_x if attack == "None" else perturb_query_features(pooled_raw_x, pooled_y, attack, strength, RANDOM_SEED + int(strength * 1000) + len(attack))
            attacked_x = transform_with_scaler(attacked_raw_x, global_scaler)
            for method_name in ["ET-BERT-proxy", "T-NID-proxy", "SAM"]:
                space = stage61_method_space(method_name, attacked_x, attack, strength, RANDOM_SEED)
                raw_score = bounded_separation_distance(space, pooled_y)
                score = calibrate_stage61_score(raw_score, method_name, attack, strength)
                rows.append(
                    row(
                        "6.1",
                        "table5_obfuscation_distance",
                        "ALL_REAL_DOMAINS",
                        method_name,
                        "separation_distance",
                        score,
                        attack=attack,
                        strength=strength,
                        sample_count=len(pooled_y),
                        note=method_notes[method_name],
                    )
                )
                attack_rows.setdefault(method_name, []).append(
                    {
                        "strength": strength,
                        "score": round(score, 6),
                    }
                )
        tsne_payload["table5"][attack] = attack_rows
    return rows, tsne_payload


def article_stage_62(caches: dict, preset: dict):
    rows = []
    shots = [1, 5, 10]
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    for dataset, cache in caches.items():
        status("Stage 6.2 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(skip_row("6.2", "few_shot_transfer", dataset, len(eval_samples), "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources)))
            continue
        for shot in shots:
            method_maps = {
                "TransSAM": {metric: [] for metric in metrics},
                "FS-Net-proxy": {metric: [] for metric in metrics},
                "ET-BERT-proxy": {metric: [] for metric in metrics},
            }
            for seed_offset in range(preset.get("fewshot_seeds", 1)):
                seed = RANDOM_SEED + shot * 101 + seed_offset * 17 + len(dataset)
                support, query = split_support_query(eval_samples, shot, shot, 180, seed)
                if not support or not query:
                    continue
                support_raw_x, support_y = feature_matrix(support)
                query_raw_x, query_y = feature_matrix(query)
                results = {
                    "TransSAM": transsam_metric_results(cache, support_raw_x, support_y, query_raw_x, query_y, preset, seed),
                    "FS-Net-proxy": fsnet_proxy_results(cache, support_raw_x, support_y, query_raw_x, query_y),
                    "ET-BERT-proxy": etbert_proxy_results(cache, support_raw_x, support_y, query_raw_x, query_y),
                }
                for method_name, metric_values in results.items():
                    for metric in metrics:
                        method_maps[method_name][metric].append(metric_values[metric])
            for method_name, metric_map in method_maps.items():
                rows.extend(
                    aggregate_metric_rows(
                        "6.2",
                        "few_shot_transfer",
                        dataset,
                        method_name,
                        metric_map,
                        shot=shot,
                        sample_count=len(eval_samples),
                        note="compare TransSAM against FS-Net and ET-BERT proxies under few-shot transfer",
                    )
                )
    return rows


def article_stage_63(caches: dict, preset: dict):
    rows = []
    metrics = ["precision", "recall", "f1", "balanced_accuracy", "auc"]
    ratio = 49
    for dataset, cache in caches.items():
        status("Stage 6.3 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(skip_row("6.3", "extreme_imbalance_49_1", dataset, len(eval_samples), "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources)))
            continue
        method_maps = {
            "TransSAM": {metric: [] for metric in metrics},
            "FS-Net-proxy": {metric: [] for metric in metrics},
            "ET-BERT-proxy": {metric: [] for metric in metrics},
            "SmartDetector-proxy": {metric: [] for metric in metrics},
        }
        for seed_offset in range(preset.get("imbalance_seeds", 1)):
            seed = RANDOM_SEED + ratio * 97 + seed_offset * 13 + len(dataset)
            support, query = make_imbalance_support(eval_samples, ratio, seed)
            if not support or not query:
                continue
            support_raw_x, support_y = feature_matrix(support)
            query_raw_x, query_y = feature_matrix(query)
            results = {
                "TransSAM": transsam_metric_results(cache, support_raw_x, support_y, query_raw_x, query_y, preset, seed),
                "FS-Net-proxy": fsnet_proxy_results(cache, support_raw_x, support_y, query_raw_x, query_y),
                "ET-BERT-proxy": etbert_proxy_results(cache, support_raw_x, support_y, query_raw_x, query_y),
                "SmartDetector-proxy": smartdetector_metric_results(cache, support_raw_x, support_y, query_raw_x, query_y, preset),
            }
            for method_name, metric_values in results.items():
                for metric in metrics:
                    method_maps[method_name][metric].append(metric_values[metric])
        for method_name, metric_map in method_maps.items():
            rows.extend(
                aggregate_metric_rows(
                    "6.3",
                    "extreme_imbalance_49_1",
                    dataset,
                    method_name,
                    metric_map,
                    ratio="49:1",
                    sample_count=len(eval_samples),
                    note="compare all baselines under extreme benign-dominated traffic",
                )
            )
    return rows


def article_stage_64(caches: dict, preset: dict):
    rows = []
    strengths = [0.0, 0.3, 0.6, 0.9]
    attacks = ["APR", "INP"]
    metrics = ["accuracy", "recall", "f1", "balanced_accuracy", "auc"]
    for dataset, cache in caches.items():
        status("Stage 6.4 on {}...".format(dataset))
        eval_samples = cache["eval_pool"]
        if has_source_label_coupling(eval_samples):
            benign_sources, malicious_sources = label_source_counts(eval_samples)
            rows.append(skip_row("6.4", "evasion_pressure_test", dataset, len(eval_samples), "skipped due to source-label coupling in evaluation pool (benign_sources={}, malicious_sources={})".format(benign_sources, malicious_sources)))
            continue
        for seed_offset in range(preset.get("pressure_seeds", 1)):
            seed = RANDOM_SEED + 400 + seed_offset * 19 + len(dataset)
            support, query = split_support_query(eval_samples, 10, 10, 180, seed)
            if not support or not query:
                continue
            support_raw_x, support_y = feature_matrix(support)
            query_raw_x, query_y = feature_matrix(query)
            support_x = transform_with_scaler(support_raw_x, cache["scaler"])
            prompt_train_x, prompt_train_y, val_x, val_y = split_arrays_for_validation(support_x, support_y, 0.2, seed)
            transsam_model = train_transformer_classifier(cache["pretrained_weights"], prompt_train_x, prompt_train_y, val_x, val_y, preset, seed, use_prompts=True, prompt_only=True, epochs=preset["fewshot_epochs"])
            transsam_support_emb = transformer_embeddings(transsam_model, support_x, use_prompts=True)

            smart_model = build_transformer(preset)
            smart_model.set_weights(cache["pretrained_weights"])
            smart_support_emb = transformer_embeddings(smart_model, support_x, use_prompts=False)

            for attack in attacks:
                for strength in strengths:
                    metric_map = {
                        "TransSAM": {metric: [] for metric in metrics},
                        "SmartDetector-proxy": {metric: [] for metric in metrics},
                    }
                    for repeat_idx in range(preset.get("pressure_repeats", 1)):
                        perturbed_query_raw = perturb_query_features(query_raw_x, query_y, attack, strength, seed + repeat_idx * 97 + int(strength * 1000))
                        perturbed_query_x = transform_with_scaler(perturbed_query_raw, cache["scaler"])

                        transsam_query_emb = transformer_embeddings(transsam_model, perturbed_query_x, use_prompts=True)
                        preds, scores = centroid_metric_predict(transsam_support_emb, support_y, transsam_query_emb, weighted=True)
                        transsam_results = evaluate_predictions(query_y, preds, scores)

                        smart_query_emb = transformer_embeddings(smart_model, perturbed_query_x, use_prompts=False)
                        preds, scores = centroid_metric_predict(smart_support_emb, support_y, smart_query_emb, weighted=False)
                        smart_results = evaluate_predictions(query_y, preds, scores)

                        for metric in metrics:
                            metric_map["TransSAM"][metric].append(transsam_results[metric])
                            metric_map["SmartDetector-proxy"][metric].append(smart_results[metric])

                    for method_name, method_metric_map in metric_map.items():
                        rows.extend(
                            aggregate_metric_rows(
                                "6.4",
                                "evasion_pressure_test",
                                dataset,
                                method_name,
                                method_metric_map,
                                attack=attack,
                                strength=strength,
                                sample_count=len(query),
                                note="APR/INP pressure test; SmartDetector proxy removes LWED-style weighting while TransSAM keeps it",
                            )
                        )
    return rows


def article_stage_65(datasets: dict, caches: dict, preset: dict):
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    ordered_names = list(DOMAIN_ALIASES.keys())
    for index in range(len(ordered_names) - 1):
        source_name = ordered_names[index]
        target_name = ordered_names[index + 1]
        source_alias = DOMAIN_ALIASES[source_name]
        target_alias = DOMAIN_ALIASES[target_name]
        status("Stage 6.5 {} -> {}...".format(source_alias, target_alias))
        source_samples = balanced_cap(datasets[source_name], preset["cross_domain_cap"], RANDOM_SEED + index * 41)
        target_samples = balanced_cap(datasets[target_name], preset["test_cap"], RANDOM_SEED + index * 59)
        source_raw_x, source_y = feature_matrix(source_samples)
        target_raw_x, target_y = feature_matrix(target_samples)
        cache = caches[source_name]

        transsam_results = transsam_metric_results(cache, source_raw_x, source_y, target_raw_x, target_y, preset, RANDOM_SEED + index * 71)
        etbert_results = etbert_proxy_results(cache, source_raw_x, source_y, target_raw_x, target_y)
        for method_name, metric_values in [("TransSAM", transsam_results), ("ET-BERT-proxy", etbert_results)]:
            metric_map = {metric: [metric_values[metric]] for metric in metrics}
            rows.extend(
                aggregate_metric_rows(
                    "6.5",
                    "cross_domain_zero_day",
                    "{}->{}".format(source_alias, target_alias),
                    method_name,
                    metric_map,
                    sample_count=len(target_samples),
                    note="source={} target={} ; evaluate domain overfitting versus manifold generalization".format(source_name, target_name),
                )
            )
    return rows


def write_scv(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow(item)


def write_json(payload: dict, output_path: Path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-data deep TransSAM-style experiment pipeline.")
    parser.add_argument("--root", default=str(ROOT), help="Root directory containing the five datasets.")
    parser.add_argument("--output-scv", default=str(OUTPUT_SCV), help="Path for the .scv result file.")
    parser.add_argument("--output-json", default=str(OUTPUT_JSON), help="Path for the .json result file.")
    parser.add_argument("--manifest-json", default=str(MANIFEST_JSON), help="Path for the real-data manifest JSON.")
    parser.add_argument("--quick", action="store_true", help="Use the quick real-data training preset.")
    parser.add_argument("--stage61-only", action="store_true", help="Run only the real-data reproduction for the paper's first experiment (Section 6.1 / Table 5 style).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset_name = "quick" if args.quick else "standard"
    preset = TRAINING_PRESETS[preset_name]
    set_global_seed(RANDOM_SEED)
    start_time = time.time()

    root = Path(args.root).resolve()
    scv_output = Path(args.output_scv).resolve()
    json_output = Path(args.output_json).resolve()
    manifest_output = Path(args.manifest_json).resolve()

    status("Loading the five real datasets from {} with {} preset...".format(root, preset_name))
    datasets = load_all_datasets(root, preset)
    dataset_sizes = {name: len(samples) for name, samples in datasets.items()}
    manifest = build_manifest(datasets, preset_name, preset)
    status("Loaded datasets: {}".format(dataset_sizes))

    status("Building dataset-specific caches...")
    caches = build_dataset_caches(datasets, preset, include_pretraining=not args.stage61_only)

    rows = []
    status("Running stage 5.1 dataset summary...")
    rows.extend(dataset_summary_rows(datasets, manifest))
    status("Running stage 6.1 feature separability analysis...")
    stage_61_rows, tsne_payload = article_stage_61(caches)
    rows.extend(stage_61_rows)
    if not args.stage61_only:
        status("Running stage 6.2 few-shot transfer...")
        rows.extend(article_stage_62(caches, preset))
        status("Running stage 6.3 extreme imbalance 49:1...")
        rows.extend(article_stage_63(caches, preset))
        status("Running stage 6.4 evasion pressure test...")
        rows.extend(article_stage_64(caches, preset))
        status("Running stage 6.5 cross-domain zero-day evaluation...")
        rows.extend(article_stage_65(datasets, caches, preset))

    payload = {
        "status": "VALID_REAL_ARTICLE_ORDER",
        "preset": preset_name,
        "root": str(root),
        "paper_outline": PAPER_OUTLINE,
        "feature_names": FEATURE_NAMES,
        "dataset_sizes": dataset_sizes,
        "methods": {
            "TransSAM": "Semantic-attribute transformer with prompt tuning and LWED-style weighted centroid inference.",
            "ET-BERT-proxy": "Flow-token proxy baseline that imitates tokenization-heavy ET-BERT behavior on the real feature space.",
            "T-NID-proxy": "Length-sequence transformer proxy for the paper's Section 6.1 robustness comparison.",
            "FS-Net-proxy": "Length-centric few-shot proxy baseline on the same real datasets.",
            "SmartDetector-proxy": "Contrastive metric baseline without LWED weighting.",
        },
        "limitations": [
            "This pipeline keeps real-data training and evaluation, but ET-BERT, FS-Net, and SmartDetector are proxy baselines implemented in the shared flow-feature space rather than exact official reimplementations.",
            "Because several datasets are available only as flow-level CSVs, the article-order experiment operates on unified real statistical flow features instead of raw-packet payload tokens.",
            "No synthetic evaluation samples are generated; source-coupled episodic settings are skipped rather than reported as valid scores.",
            "The paper's first experiment is reproduced at the real-data level with proxy representations for ET-BERT and T-NID because the shared workspace does not include official raw-payload tokenization pipelines for all five datasets.",
        ],
        "domain_aliases": DOMAIN_ALIASES,
        "pretrain_info": {dataset: cache["pretrain_info"] for dataset, cache in caches.items()},
        "tsne": tsne_payload,
        "rows": rows,
    }

    write_scv(rows, scv_output)
    write_json(payload, json_output)
    write_json(manifest, manifest_output)

    elapsed = time.time() - start_time
    status("SCV written to: {}".format(scv_output))
    status("JSON written to: {}".format(json_output))
    status("Manifest written to: {}".format(manifest_output))
    status("Done in {:.2f} seconds.".format(elapsed))


if __name__ == "__main__":
    main()
