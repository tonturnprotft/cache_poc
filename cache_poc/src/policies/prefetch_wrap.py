# src/policies/prefetch_wrap.py
import torch
import collections


class PrefetchWrapper:
    """
    Wrap an inner byte-aware cache policy and *selectively* prefetch items
    predicted by an LSTM model.

    Parameters
    ----------
    inner_policy : object
        Already-constructed cache policy with .request() and .resize().
    model : torch.nn.Module
        Trained next-item predictor. Must output logits over item vocab.
    idx2vid : dict[int, str]
        Reverse mapping from token index -> original video id.
    topN : int
        Max number of predictions to prefetch per trigger.
    win : int
        History length (sequence window) stored per user.
    prob_thresh : float
        Minimum softmax probability of top-1 to trigger prefetch.
    budget_ratio : float
        Max fraction (0–1) of cache capacity reserved for prefetched bytes.
        Prevents cache pollution.
    """

    def __init__(self, inner_policy, model, idx2vid,
                 topN: int = 1, win: int = 5,
                 prob_thresh: float = 0.25,
                 budget_ratio: float = 0.10):
        self.inner = inner_policy
        self.model = model.eval()
        self.idx2vid = idx2vid

        self.topN = topN
        self.win = win
        self.prob_th = prob_thresh
        self.budget_ratio = budget_ratio

        # per-user recent history of vid indices
        self.hist = collections.defaultdict(list)

        # bytes we inserted via prefetch (for budget control)
        self.prefetch_bytes = 0
        self._update_budget_bytes()

    # ------------------------------------------------------------------
    def _update_budget_bytes(self):
        cap = getattr(self.inner, 'cap',
                      getattr(self.inner, 'capacity', None))
        self.budget_bytes = int(cap * self.budget_ratio) if cap else 0

    # ------------------------------------------------------------------
    def request(self, key, size, ts, user=None, vid_idx=None):
        # normal request goes to inner policy
        hit = self.inner.request(key, size, ts)

        # record user history
        if user is not None and vid_idx is not None:
            h = self.hist[user]
            h.append(vid_idx)
            if len(h) > self.win:
                del h[:-self.win]

        # Only consider prefetch on MISS and if we have enough context
        if (not hit) and user is not None and len(self.hist[user]) == self.win:
            with torch.no_grad():
                x = torch.tensor([self.hist[user]], dtype=torch.long)
                logits = self.model(x)[0]
                probs = torch.softmax(logits, dim=-1)
                topk = torch.topk(probs, self.topN)
                top_probs = topk.values.tolist()
                top_idxs = topk.indices.tolist()

            # Trigger only if confident enough
            if top_probs and top_probs[0] >= self.prob_th:
                ladder = key.split('_')[-1]  # keep same resolution tier
                for p, idx in zip(top_probs, top_idxs):
                    if p < self.prob_th:
                        continue
                    vid = self.idx2vid[idx]
                    pre_key = f"{vid}_{ladder}"

                    # Respect prefetch budget
                    cap_attr = getattr(self.inner, 'cap',
                                       getattr(self.inner, 'capacity', None))
                    size_attr = getattr(self.inner, 'size', None)
                    if cap_attr is not None and size_attr is not None:
                        if self.prefetch_bytes + size > self.budget_bytes:
                            break

                    inserted_hit = self.inner.request(pre_key, size, ts)
                    if not inserted_hit:
                        self.prefetch_bytes += size

        return hit

    # ------------------------------------------------------------------
    def resize(self, new_cap):
        # propagate resize
        if hasattr(self.inner, 'resize'):
            self.inner.resize(new_cap)
        # recompute budget
        self._update_budget_bytes()
        # if we shrank hard, just cap our accounting
        if self.prefetch_bytes > self.budget_bytes:
            self.prefetch_bytes = self.budget_bytes