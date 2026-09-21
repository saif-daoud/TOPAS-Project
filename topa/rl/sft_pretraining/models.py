from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel

class RobertaForConvStateMultiHead(RobertaPreTrainedModel):
    """Multi-head classification: one head per conversation-state dimension."""

    def __init__(self, config, dim_num_labels: Dict[str, int]):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dim_names = list(dim_num_labels.keys())
        # PEFT wraps modules listed in `modules_to_save` and rejects ModuleDict.
        # Using ModuleList keeps the model simple and lets us save the heads via
        # modules_to_save=["heads.<i>", ...].
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, dim_num_labels[d]) for d in self.dim_names])
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None, # this is for TaskType.FEATURE_EXTRACTION
        labels=None,
        **kwargs, # to ignore extra keys from wrappers
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,  # pass through
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0]  # [B, H]
        cls = self.dropout(cls)

        logits_list = [head(cls) for head in self.heads]
        loss = None
        if labels is not None:
            # Average CE over dims.
            ce = nn.CrossEntropyLoss()
            losses = []
            for i, logits in enumerate(logits_list):
                losses.append(ce(logits, labels[:, i]))
            loss = torch.stack(losses).mean()

        return {"loss": loss, "logits": logits_list}

class RobertaForMicroActionMultiHead(RobertaPreTrainedModel):
    """Shared Roberta backbone + one classification head per macro action."""

    def __init__(self, config, macro_id_to_num_micro: Dict[int, int]):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Store as string keys because ModuleDict expects str keys.
        self.heads = nn.ModuleDict({str(mid): nn.Linear(config.hidden_size, n) for mid, n in macro_id_to_num_micro.items()})
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        macro_action_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # macro_action_id: (batch,) ints that select the head
        # labels: (batch,) ints inside the selected head's micro action space
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss()
            losses = []
            for mid in torch.unique(macro_action_id).tolist():
                mask = (macro_action_id == mid)
                logits = self.heads[str(mid)](cls[mask])
                losses.append(ce(logits, labels[mask]))
            loss = torch.stack(losses).mean()

        return {"loss": loss, "cls": cls}

class RobertaForMacroActionFusion(RobertaPreTrainedModel):
    """Macro-action classification with late fusion:

    - Encode dialogue with RoBERTa (CLS feature)
    - Encode conversation-state labels as a concatenated one-hot vector
    - Merge (concat) RoBERTa features + projected conv-state vector
    """

    def __init__(self, config, conv_state_dim: int):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conv_state_dim = int(conv_state_dim)
        if self.conv_state_dim <= 0:
            raise ValueError("conv_state_dim must be > 0 for RobertaForMacroActionFusion")

        # Project conv_state_vec into the same space as the CLS embedding.
        self.conv_proj = nn.Linear(self.conv_state_dim, config.hidden_size)

        # Late-fusion classifier.
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        conv_state_vec: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # NOTE: PEFT wrappers may always pass `inputs_embeds` (even if None). We accept it
        # for compatibility and pass through to the underlying RoBERTa module.
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])

        if conv_state_vec is None:
            conv_state_vec = torch.zeros((cls.shape[0], self.conv_state_dim), device=cls.device, dtype=cls.dtype)
        conv_state_vec = conv_state_vec.to(device=cls.device, dtype=cls.dtype)
        cs = self.dropout(self.conv_proj(conv_state_vec))

        fused = torch.cat([cls, cs], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}