"""One architecture-only finite control over the pinned shape probe."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import run_paired_shape_train_probe as runner
from prepare_paired_port_semantics import ROOT,load,save,sha


def fit_head(features,labels):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    head=torch.nn.Sequential(torch.nn.Linear(features.shape[1],32),torch.nn.GELU(),torch.nn.Linear(32,3))
    torch.nn.init.zeros_(head[-1].weight);torch.nn.init.zeros_(head[-1].bias)
    counts=torch.bincount(labels,minlength=3).float()
    if (counts==0).any():raise ValueError('Missing training class')
    weights=counts.sum()/(3*counts)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    for step in range(400):
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(head(features),labels,weight=weights)
        if not torch.isfinite(loss):raise ValueError('Nonfinite nonlinear training loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Invalid nonlinear gradients')
        optimizer.step()
    return head.eval()


def main():
    import torch
    torch.set_num_threads(1)
    old_out,old_fit=runner.OUT,runner.fit_head
    destination=ROOT/'artifacts/paired_shape_nonlinear_20261003'
    pins={str(p):sha(p) for p in (Path(__file__),Path(runner.__file__),ROOT/'artifacts/paired_shape_nonlinear_preregistration_20261003/PLAN.md')}
    try:
        smoke=torch.arange(18*6147,dtype=torch.float32).reshape(18,6147)/100000.
        head=fit_head(smoke,torch.arange(18)%3)
        assert all(torch.isfinite(p).all() for p in head.parameters())
        runner.OUT=destination;runner.fit_head=fit_head;runner.main()
        assert {p:sha(Path(p)) for p in pins}==pins
        save(destination/'architecture_protocol.json',dict(pins=pins,architecture=[6147,32,'GELU',3],fixed_steps=400,
            smoke_finite_gradients=True,final_zero_initialization=True,no_encoder_inference=True,no_release_changes=True,no_deployment=True))
    finally:runner.OUT,runner.fit_head=old_out,old_fit

if __name__=='__main__':main()
