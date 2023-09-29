from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple, NewType, Any
import torch
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)

@dataclass
class DataCollatorForContrastivePreTraining:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        common_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        mlm_features = []  # we only get the original sample for mlm

        for feature in features:
            mlm_dict = dict()
            for k in feature:
                if k in common_keys:
                    mlm_dict[k] = feature[k][0]
            mlm_dict["token_type_ids"] = feature["tree_labels"]  # since mlm and tree recovery both only apply to the original samples, we process them together
            mlm_features.append(mlm_dict)

            for i in range(num_sent):
                flat_dict = dict()
                for k in feature:
                    if k in common_keys:
                        flat_dict[k] = feature[k][i]
                flat_features.append(flat_dict)
                # flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        mlm_batch = self.tokenizer.pad(
            mlm_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        tree_labels = mlm_batch["token_type_ids"]
        mlm_attention_mask = mlm_batch["attention_mask"]
        mlm_input_ids, mlm_labels, masked_indices, indices_replaced, indices_random = self.mask_tokens(mlm_batch["input_ids"])

        batch = {k: batch[k].view(bs, num_sent, -1) if k in common_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        tree_labels[tree_labels == 0] = -100 # since we use tree_labels as token_type_ids, they use 0 rather than pad_token to do padding

        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "mlm_input_ids": mlm_input_ids,
                "mlm_attention_mask": mlm_attention_mask, "mlm_labels": mlm_labels, "tree_labels": tree_labels}
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices, indices_replaced, indices_random
