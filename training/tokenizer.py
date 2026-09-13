from __future__ import annotations

import re

import torch


DEFAULT_WORDS = (
    "pick", "up", "the", "red", "green", "blue", "block", "cube", "and",
    "put", "place", "move", "it", "in", "inside", "yellow", "target", "area",
)


class WordTokenizer:
    """A visible word tokenizer for the small command language used by 3jsVLA."""

    def __init__(self, extra_words: tuple[str, ...] = ()) -> None:
        words = ("<pad>", "<bos>", "<eos>", "<unk>") + DEFAULT_WORDS + extra_words
        self.words = tuple(dict.fromkeys(words))
        self.token_to_id = {word: index for index, word in enumerate(self.words)}

    def encode(self, text: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens)
        ids.append(self.token_to_id["<eos>"])
        ids = ids[:length]
        mask = [True] * len(ids)
        ids.extend([self.token_to_id["<pad>"]] * (length - len(ids)))
        mask.extend([False] * (length - len(mask)))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def batch(self, texts: list[str], length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text, length) for text in texts]
        ids = torch.stack([item[0] for item in encoded]).to(device)
        mask = torch.stack([item[1] for item in encoded]).to(device)
        return ids, mask
