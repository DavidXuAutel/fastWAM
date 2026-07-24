from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Dataset


class WeightedSourceDataset(Dataset):
    """Randomly sample one of several datasets using fixed source probabilities."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        probs: Sequence[float],
        names: Sequence[str],
        generator: torch.Generator | None = None,
    ):
        if not (len(datasets) == len(probs) == len(names)):
            raise ValueError("datasets, probs, and names must have equal lengths")
        if not datasets:
            raise ValueError("at least one source dataset is required")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("source datasets must not be empty")
        if any(probability < 0 for probability in probs):
            raise ValueError("source probabilities must be non-negative")
        if abs(sum(probs) - 1.0) >= 1e-6:
            raise ValueError("source probabilities must sum to 1")
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")

        self.datasets = list(datasets)
        self.probs = torch.tensor(probs, dtype=torch.float64)
        self.names = list(names)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(0)
        self.counts = {name: 0 for name in self.names}

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, _index):
        source_index = int(torch.multinomial(self.probs, 1, generator=self.generator).item())
        self.counts[self.names[source_index]] += 1
        sample_index = int(
            torch.randint(
                len(self.datasets[source_index]),
                (1,),
                generator=self.generator,
            ).item()
        )
        sample = self.datasets[source_index][sample_index]
        if not isinstance(sample, Mapping):
            raise TypeError("source datasets must return mapping samples")
        result = dict(sample)
        result["data_source"] = self.names[source_index]
        return result

    def pop_source_counts(self) -> dict[str, int]:
        counts = self.counts.copy()
        self.counts = {name: 0 for name in self.names}
        return counts
