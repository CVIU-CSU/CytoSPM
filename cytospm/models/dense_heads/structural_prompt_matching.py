# Copyright (c) The CytoSPM Authors. All rights reserved.
"""Structural prompt matching (Stage II of CytoSPM)."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralPromptMatching(nn.Module):
    """Match region features against a per-class structured prompt bank.

    For every class ``c`` with prompts ``b_1^c ... b_M^c`` this computes

    * the fine-grained attribute score ``max_j <v, b_j^c>`` over the selected
      prompts, and
    * the structural prototype score ``<v, p_c>``, where ``p_c`` is an
      image-conditioned prototype aggregated from the selected prompts.

    Prompt selection is mutual: the Top-P regions that best match class ``c``
    vote for the prompts they support, and the Top-Q supported prompts are kept.
    The selection path is detached, so gradients only flow through the scores.

    The module has no learnable parameters, so the number of prompts per class
    may differ between the base and novel vocabularies.

    Args:
        top_p (float): Regions kept per level. A value in (0, 1) is a fraction
            of the feature-map locations; a value >= 1 is a fixed count.
        top_q (float): Prompts kept per class. A value in (0, 1) is a fraction
            of that class's valid prompts; a value >= 1 is a fixed count;
            0 keeps all prompts.
        tau (float): Softmax temperature used to aggregate the prototype.
        chunk (int): Number of classes processed at once (bounds memory).
        use_prototype (bool): Whether to compute the prototype score.
        bank_mask (Tensor, optional): ``[C, M]`` bool mask of valid prompts.
    """

    def __init__(self,
                 top_p: float = 0.01,
                 top_q: float = 0.0,
                 tau: float = 0.1,
                 chunk: int = 8,
                 use_prototype: bool = True,
                 bank_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.top_p = float(top_p)
        self.top_q = float(top_q)
        self.tau = float(tau)
        self.chunk = int(chunk)
        self.use_prototype = bool(use_prototype)
        if bank_mask is not None:
            self.register_buffer(
                'bank_mask', bank_mask.bool().contiguous(), persistent=False)
        else:
            self.bank_mask = None

    def forward(self, x: torch.Tensor, prompt_bank: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (Tensor): Normalized region features ``[B, D, H, W]``.
            prompt_bank (Tensor): Prompt embeddings ``[C, M, D]``; padded
                rows are zero.

        Returns:
            tuple: Attribute score ``[B, C, H, W]`` and prototype score
            ``[B, C, H, W]`` (``None`` if ``use_prototype`` is False).
        """
        B, D, H, W = x.shape
        N = H * W
        bt = F.normalize(prompt_bank.to(x.dtype), dim=-1)
        C, M, _ = bt.shape
        mask = self.bank_mask
        if mask is None:
            mask = torch.ones(C, M, dtype=torch.bool, device=x.device)
        mask = mask.to(x.device)
        xf = x.reshape(B, D, N)
        tp = self.top_p
        P = min(int(round(tp)), N) if tp >= 1 else min(max(int(tp * N), 1), N)
        cs = self.chunk if self.chunk > 0 else C
        neg = -1e4
        s3_chunks, s2_chunks = [], []
        for c0 in range(0, C, cs):
            c1 = min(c0 + cs, C)
            Cc = c1 - c0
            bc = bt[c0:c1]                                          # [Cc, M, D]
            mc = mask[c0:c1]                                        # [Cc, M]
            L = torch.einsum('bdn,cmd->bcmn', xf, bc)               # [B, Cc, M, N]
            L = L.masked_fill((~mc).view(1, Cc, M, 1), neg)
            # Mutual selection (detached): Top-P regions -> prompt support.
            L_sel = L.detach()
            a = L_sel.amax(dim=2)                                   # [B, Cc, N]
            idx = a.topk(P, dim=2).indices                          # [B, Cc, P]
            Lsel = torch.gather(L_sel, 3, idx.unsqueeze(2).expand(-1, -1, M, -1))
            r = Lsel.amax(dim=3)                                    # [B, Cc, M]
            qf = self.top_q
            if qf >= 1:
                q = min(int(round(qf)), M)
                qidx = r.topk(q, dim=2).indices
                qmask = torch.zeros(B, Cc, M, dtype=torch.bool, device=x.device)
                qmask.scatter_(2, qidx, True)
                qmask = qmask & mc.view(1, Cc, M)
            elif 0 < qf < 1:
                # Per-class fraction of the valid prompts (padding ranks last).
                n_valid = mc.sum(dim=1)
                q_c = (qf * n_valid.float()).round().clamp(min=1).long()
                ranks = r.argsort(dim=2, descending=True).argsort(dim=2)
                qmask = (ranks < q_c.view(1, Cc, 1)) & mc.view(1, Cc, M)
            else:
                qmask = mc.view(1, Cc, M).expand(B, -1, -1)
            r = r.masked_fill(~qmask, neg)
            s3 = L.masked_fill(~qmask.unsqueeze(-1), neg).amax(dim=2)
            s3_chunks.append(s3.reshape(B, Cc, H, W))
            if not self.use_prototype:
                continue
            # Image-conditioned structural prototype from the selected prompts.
            alpha = (r / self.tau).softmax(dim=2)                   # [B, Cc, M]
            p = F.normalize(torch.einsum('bcm,cmd->bcd', alpha, bc), dim=-1)
            s2_chunks.append(torch.einsum('bdn,bcd->bcn', xf, p).reshape(B, Cc, H, W))
        s2 = torch.cat(s2_chunks, dim=1) if s2_chunks else None
        return torch.cat(s3_chunks, dim=1), s2
