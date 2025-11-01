import csv
from PIL import Image

class SimpleDatasetLoader:
    """Loads images and labels from a manifest (CSV or JSON)."""

    def __init__(self, manifest_csv=None, manifest_json=None):
        if manifest_csv is None and manifest_json is None:
            raise ValueError('Provide manifest_csv or manifest_json')
        self.entries = []
        if manifest_csv:
            with open(manifest_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    self.entries.append(r)
        else:
            import json
            with open(manifest_json, 'r', encoding='utf-8') as f:
                self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        img_path = item['image']
        label = item.get('label')
        img = Image.open(img_path).convert('RGB')
        return img, label

    def iter_batches(self, batch_size=8):
        batch = []
        labels = []
        for img, lbl in (self[i] for i in range(len(self))):
            batch.append(img)
            labels.append(lbl)
            if len(batch) == batch_size:
                yield batch, labels
                batch = []
                labels = []
        if batch:
            yield batch, labels
