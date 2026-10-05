"""fastai's learning-rate finder for our Lightning models (fastai/callback/schedule.py: LRFinder, lr_find, valley).

Exponential sweep from start_lr to end_lr over num_it training batches with AdamW, loss smoothed as fastai's Recorder
does (beta = 0.98, debiased), stop once the smoothed loss exceeds 4x its best, drop the first num_it // 10 and the last
5 points, and suggest the learning rate in the longest valley. The model's weights are restored afterwards.
"""
import copy

import torch


def valley(lrs, losses):
    """Suggests a learning rate from the longest valley (the logic of fastai's `valley`)."""
    n = len(losses)
    max_start, max_end = 0, 0
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]
    sections = (max_end - max_start) / 3
    return float(lrs[max_start + int(sections) + int(sections / 2)])


def lr_find(model, loader, weight_decay=1e-2, start_lr=1e-7, end_lr=10.0, num_it=100, beta=0.98, precision="32-true"):
    """Return (suggested learning rate, swept learning rates, smoothed losses)."""
    device = next(model.parameters()).device
    state = copy.deepcopy(model.state_dict())
    # fastai's default optimizer: Adam(mom=0.9, sqr_mom=0.99, eps=1e-5) with decoupled weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=weight_decay)
    autocast = precision.startswith("bf16") and device.type == "cuda"
    model.train()
    lrs, losses, average, best = [], [], 0.0, float("inf")
    batches = iter(loader)
    for it in range(num_it):
        lr = start_lr * (end_lr / start_lr) ** (it / num_it)
        for group in optimizer.param_groups:
            group["lr"] = lr
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        x, coords, y = model.split_batch([t.to(device) for t in batch])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred, l2 = model.predict(x, coords)
        loss = model.data_loss(pred.float(), y) + l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average = beta * average + (1 - beta) * loss.item()
        smooth = average / (1 - beta ** (it + 1))
        lrs.append(lr)
        losses.append(smooth)
        best = min(best, smooth)
        if smooth > 4 * best:
            break
    model.load_state_dict(state)
    kept_lrs, kept_losses = lrs[num_it // 10:-5], losses[num_it // 10:-5]
    if len(kept_lrs) < 3:  # diverged almost at once: fall back to the middle of the sweep
        return float(lrs[len(lrs) // 2]), lrs, losses
    return valley(kept_lrs, kept_losses), lrs, losses
