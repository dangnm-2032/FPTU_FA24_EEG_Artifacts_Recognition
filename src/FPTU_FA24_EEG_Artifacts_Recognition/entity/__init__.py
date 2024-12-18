from dataclasses import dataclass
from pathlib import Path

@dataclass
class EEGModule:
    nb_classes: int
    Chans: int
    Samples: int
    dropoutRate: float
    kernLength: int
    F1: int
    D: int
    F2: int
    dropoutType: str

@dataclass
class TrainingParams:
    learning_rate: float
    batch_size: int
    epochs: int

@dataclass
class EEGModel:
    right: EEGModule
    left: EEGModule
    teeth: EEGModule
    both: EEGModule
    eyebrows: EEGModule
    training: TrainingParams

@dataclass
class RecordSubject:
    id: int
    name: str
    position: int
    trial: int

@dataclass
class EEGDataset:
    raw_data_path: Path
    raw_roi_path: Path
    output_data_path: Path
    output_roi_path: Path
    filepath_format: str
    label: list
    details: dict
    scaler_path: Path
    scaler_extension: str
    skip_preprocess_data: bool
    save_test_data: Path

@dataclass
class EEGModelConfig:
    save_path: Path
    save_name: str
    weight_extension: str
    config_extension: str
    history_extension: str
    inference_model: Path