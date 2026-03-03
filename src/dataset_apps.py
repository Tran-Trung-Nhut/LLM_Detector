import random
from PIL import Image
from torch.utils.data import Dataset

class AppsSingleImageDataset(Dataset):
    def __init__(self, rows, image_strategy="first", seed=42):
        self.rows = rows
        self.image_strategy = image_strategy
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def pick_image(self, image_paths):
        if not image_paths:
            return None
        if self.image_strategy == "first":
            return image_paths[0]
        if self.image_strategy == "random":
            return self.rng.choice(image_paths)
        raise ValueError(f"Unknown image_strategy={self.image_strategy}")

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = self.pick_image(r["image_paths"])
        image = Image.open(img_path).convert("RGB")
        return {
            "app_id": r["app_id"],
            "text": r["text"],
            "label_binary": int(r["label_binary"]),
            "label_3class": r.get("label_3class"),
            "image": image,
        }