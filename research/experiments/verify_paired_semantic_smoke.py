"""Real paired features gradient smoke plus real original SIFT cache parity."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,DATA,load,save,sha,read_image,REFERENCE_SHA
from paired_port_semantics import CachedReferenceSIFT


def main():
    destination=OUT/'smoke/verification.json'
    if destination.exists():raise FileExistsError('Preserve smoke verification')
    report=load(OUT/'smoke/report.json');assert report['status']=='complete' and report['gt_valid']==report['gt_targets']
    assert {p:sha(Path(p)) for p in report['pins']}==report['pins']
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import assembly_auto_review_robust_v3 as original
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(0)
    data=torch.load(OUT/'smoke/features.pt',map_location='cpu',weights_only=True)
    assert sha(OUT/'smoke/features.pt')==report['aggregate_feature_sha256']
    head=torch.nn.Linear(6144,3);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    before={k:v.detach().clone() for k,v in head.state_dict().items()};optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    counts=torch.bincount(data['labels'],minlength=3).float();assert (counts>0).all()
    value=F.cross_entropy(head(data['features']),data['labels'],weight=counts.sum()/(3*counts));assert torch.isfinite(value)
    value.backward();assert all(torch.isfinite(p.grad).all() for p in head.parameters());assert float(head.weight.grad.abs().sum())>0
    optimizer.step();changed=sum(not torch.equal(v,head.state_dict()[k]) for k,v in before.items());assert changed>0
    reference=read_image(DATA/'images/train01/normal_073.JPG');image=read_image(DATA/'images/train01'/report['source_groups'][0])
    cv2.setRNGSeed(0);_,native=original.automatic_homography(reference,image)
    with CachedReferenceSIFT(reference) as cache:
        cv2.setRNGSeed(0);_,first=original.automatic_homography(reference,image)
        cv2.setRNGSeed(0);_,second=original.automatic_homography(reference,image)
    assert cache.cache_hits>=1
    for candidate in (first,second):
        assert candidate['alignment_quality']['reliable'] is True
        assert candidate['matches']==native['matches'] and candidate['inliers']==native['inliers']
        np.testing.assert_allclose(candidate['source_to_reference_homography'],native['source_to_reference_homography'],rtol=0,atol=1e-8)
    assert {p:sha(Path(p)) for p in report['pins']}==report['pins']
    result=dict(status='complete',finite_gradient_step=True,changed_tensors=changed,training_loss=float(value.detach()),
        original_registration_cache_parity=True,reference_descriptor_cache_hits=cache.cache_hits,
        real_sources=report['source_groups'],no_field_accuracy=True,no_model_deployment=True)
    save(destination,result);print(str(result),flush=True)


if __name__=='__main__':main()
