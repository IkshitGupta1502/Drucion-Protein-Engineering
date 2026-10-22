"""ESM-2 sequence embeddings with on-disk caching.

The embedder mean-pools the final hidden layer over **real residue positions
only**: padding, ``<cls>`` and ``<eos>`` tokens are excluded from the average.
The model is loaded at most once per process and reused.

Nothing here ever sees a fitness label, so embeddings may be computed for the
whole candidate pool up front without leaking the oracle's held-out values.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import CACHE_DIR, resolve_device

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cache_key(model_name: str, sequences: Sequence[str], tag: str = "") -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(tag.encode())
    h.update(str(len(sequences)).encode())
    for s in sequences:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:24]


class ESM2Embedder:
    """Mean-pooled ESM-2 residue representations.

    Parameters
    ----------
    model_name:
        Any HuggingFace ESM-2 checkpoint. Defaults to the 8M-parameter model,
        which runs comfortably on CPU; larger checkpoints work unchanged.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "auto",
        batch_size: int = 16,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir or CACHE_DIR) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, EsmModel

        logger.info("Loading %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = EsmModel.from_pretrained(self.model_name)
        model.eval()
        self._model = model.to(self.device)
        self._torch = torch

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    # ------------------------------------------------------------------

    def embed_sequences(
        self, sequences: Sequence[str], use_cache: bool = True, progress=None
    ) -> np.ndarray:
        """Return ``(n_sequences, hidden_size)`` mean-pooled embeddings.

        ``progress`` is an optional ``callable(done, total)`` for UI updates.
        """
        sequences = list(sequences)
        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        cache_path = (
            self.cache_dir / f"{_cache_key(self.model_name, sequences)}.npy"
            if use_cache
            else None
        )
        if cache_path is not None and cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape[0] == len(sequences):
                logger.info("Loaded cached embeddings from %s", cache_path.name)
                if progress:
                    progress(len(sequences), len(sequences))
                return arr
            logger.warning("Ignoring stale embedding cache %s", cache_path.name)

        self._ensure_loaded()
        torch = self._torch
        out = np.empty((len(sequences), self.embedding_dim), dtype=np.float32)

        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                hidden = self._model(
                    input_ids=input_ids, attention_mask=attn
                ).last_hidden_state  # (B, L, H)

            # Build a mask over *real residues*: attended, and not a special token.
            special = torch.zeros_like(attn, dtype=torch.bool)
            for tok_id in self._tokenizer.all_special_ids:
                special |= input_ids == tok_id
            residue_mask = (attn.bool() & ~special).unsqueeze(-1).to(hidden.dtype)

            summed = (hidden * residue_mask).sum(dim=1)
            counts = residue_mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
            out[start : start + len(batch)] = pooled.float().cpu().numpy()

            if progress:
                progress(min(start + len(batch), len(sequences)), len(sequences))

        if cache_path is not None:
            np.save(cache_path, out)
            logger.info("Cached embeddings to %s", cache_path.name)
        return out

    # ------------------------------------------------------------------

    def embed_sites(
        self,
        sequences: Sequence[str],
        positions: Sequence[Sequence[int]],
        wild_type: str,
        use_cache: bool = True,
        progress=None,
    ) -> np.ndarray:
        """Site-resolved features for mutated variants.

        Mean-pooling a 250-residue protein barely moves when one residue
        changes, so whole-sequence pooling is a weak representation for single
        mutants. This method instead returns, for each variant,

        ``[ h_mut(site) , h_mut(site) - h_wt(site) ]``

        where ``h(site)`` is the final-layer hidden state at the mutated
        residue, averaged over the variant's mutated positions. The first block
        says *what the mutated site now looks like in context*; the second
        isolates *what the substitution changed*, cancelling the large shared
        component. Output dimension is ``2 * hidden_size``.

        ``positions`` are 1-indexed into ``wild_type``, matching ProteinGym.
        """
        import numpy as _np

        sequences = list(sequences)
        positions = [list(p) for p in positions]
        if len(sequences) != len(positions):
            raise ValueError("sequences and positions must be the same length")
        if not sequences:
            return _np.empty((0, 2 * self.embedding_dim), dtype=_np.float32)

        cache_path = (
            self.cache_dir
            / f"{_cache_key(self.model_name, sequences, tag='site-v1')}.npy"
            if use_cache
            else None
        )
        if cache_path is not None and cache_path.exists():
            arr = _np.load(cache_path)
            if arr.shape[0] == len(sequences):
                logger.info("Loaded cached site embeddings from %s", cache_path.name)
                if progress:
                    progress(len(sequences), len(sequences))
                return arr
            logger.warning("Ignoring stale site-embedding cache %s", cache_path.name)

        self._ensure_loaded()
        H = self.embedding_dim

        # Wild-type reference hidden states, computed once.
        wt_hidden = self._residue_hidden_states([wild_type])[0]  # (L, H)

        out = _np.empty((len(sequences), 2 * H), dtype=_np.float32)
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            batch_pos = positions[start : start + self.batch_size]
            hidden = self._residue_hidden_states(batch)  # list of (L, H)
            for j, (h, pos_list) in enumerate(zip(hidden, batch_pos)):
                if not pos_list:
                    raise ValueError("Each variant needs at least one position")
                rows = [p - 1 for p in pos_list]
                h_site = h[rows].mean(axis=0)
                h_wt_site = wt_hidden[rows].mean(axis=0)
                out[start + j] = _np.concatenate([h_site, h_site - h_wt_site])
            if progress:
                progress(min(start + len(batch), len(sequences)), len(sequences))

        if cache_path is not None:
            np.save(cache_path, out)
            logger.info("Cached site embeddings to %s", cache_path.name)
        return out

    def _residue_hidden_states(self, sequences: Sequence[str]) -> list[np.ndarray]:
        """Final-layer hidden states for real residues only, one array per sequence."""
        self._ensure_loaded()
        torch = self._torch
        enc = self._tokenizer(
            list(sequences), return_tensors="pt", padding=True, truncation=False
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            hidden = self._model(
                input_ids=input_ids, attention_mask=attn
            ).last_hidden_state

        special = torch.zeros_like(attn, dtype=torch.bool)
        for tok_id in self._tokenizer.all_special_ids:
            special |= input_ids == tok_id
        residue_mask = attn.bool() & ~special

        out: list[np.ndarray] = []
        h_cpu = hidden.float().cpu().numpy()
        mask_cpu = residue_mask.cpu().numpy()
        for row in range(h_cpu.shape[0]):
            out.append(h_cpu[row][mask_cpu[row]])
        return out

    # ------------------------------------------------------------------

    def masked_marginal_scores(
        self, wild_type: str, mutants: Sequence[str]
    ) -> np.ndarray:
        """ESM-2 masked-marginal zero-shot score for each mutant.

        For each mutated position, the wild-type residue is masked and the
        score is ``log p(mut) - log p(wt)`` summed over the mutation's
        positions — the standard pLM zero-shot fitness proxy. Used by the
        ESM-guided / evolutionary baseline; no fitness labels involved.

        Adapted in spirit from EvoEmu's ESM-2 mutation proposal module
        (github.com/ibrahimaasim77/EvoEmu), reimplemented here against
        ``EsmForMaskedLM`` with ProteinGym's 1-indexed mutation convention.
        """
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM

        from ..data.proteingym import parse_mutant

        cache_path = (
            self.cache_dir
            / f"{_cache_key(self.model_name, [wild_type, *mutants], tag='mm-v1')}.npy"
        )
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape[0] == len(mutants):
                logger.info("Loaded cached zero-shot scores from %s", cache_path.name)
                return arr

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        mlm = EsmForMaskedLM.from_pretrained(self.model_name).eval().to(self.device)

        positions = sorted(
            {s.position for m in mutants for s in parse_mutant(m)}
        )
        # log-probs at every mutated position, one masked forward pass each.
        log_probs: dict[int, torch.Tensor] = {}
        for start in range(0, len(positions), self.batch_size):
            batch_pos = positions[start : start + self.batch_size]
            seqs = []
            for pos in batch_pos:
                chars = list(wild_type)
                chars[pos - 1] = tokenizer.mask_token
                seqs.append("".join(chars))
            enc = tokenizer(seqs, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)
            with torch.no_grad():
                logits = mlm(
                    input_ids=input_ids,
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero()
            for row, col in mask_positions.tolist():
                lp = torch.log_softmax(logits[row, col], dim=-1)
                log_probs[batch_pos[row]] = lp.cpu()

        scores = np.zeros(len(mutants), dtype=float)
        for i, mutant in enumerate(mutants):
            total = 0.0
            for sub in parse_mutant(mutant):
                lp = log_probs[sub.position]
                mut_id = tokenizer.convert_tokens_to_ids(sub.mut_aa)
                wt_id = tokenizer.convert_tokens_to_ids(sub.wt_aa)
                total += float(lp[mut_id] - lp[wt_id])
            scores[i] = total
        np.save(cache_path, scores)
        return scores
