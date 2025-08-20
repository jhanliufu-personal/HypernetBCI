from torch.utils.data import Sampler
import random
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        """
        Args:
            labels (list or tensor): list of all labels in the dataset
            n_classes (int): number of classes per batch
            n_samples (int): number of samples per class
        """
        self.labels = labels
        self.labels_set = list(set(labels))
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        for label in self.labels_set:
            random.shuffle(self.label_to_indices[label])

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        # Generate batches infinitely
        while True:
            selected_classes = random.sample(self.labels_set, self.n_classes)
            batch_indices = []
            for cls in selected_classes:
                cls_indices = self.label_to_indices[cls]
                if len(cls_indices) < self.n_samples:
                    break  # Skip this iteration if not enough samples
                selected = random.sample(cls_indices, self.n_samples)
                batch_indices.extend(selected)
            if len(batch_indices) == self.batch_size:
                yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size
