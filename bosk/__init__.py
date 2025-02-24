import torch

# Определеяем устройство на котором будем производить вычисления
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Словарь классов и их численными значениями в исходном датасете
CLASS_IDS = {
    "road": 180,  # Дорога
    "lake": 117,  # Озеро / река
    "bridge": 153,  # Мост
    "tree": 106,  # Деревья
    "background": 110,  # Фон
}

NUM_CLASSES = len(CLASS_IDS)

__all__ = ["DEVICE", "CLASS_IDS", "NUM_CLASSES"]
