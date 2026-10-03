"""Only after OOF source pass, fit the fixed train192-only final linear head."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import OUT,REPO,load,save,sha
from port_semantic_verifier import fit_head
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint


def main():
    destination=OUT/'head_full'
    if destination.exists():raise FileExistsError('Preserve final head')
    scored=load(OUT/'source_train_oof/report.json');assert scored['status']=='complete' and scored['qualifies']
    prepared=load(OUT/'features_train/report.json');assert resolution_runtime_fingerprint(REPO)==prepared['runtime_fingerprint']
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_verifier.py'),
        OUT/'features_train/report.json',OUT/'features_train/features.pt',OUT/'source_train_oof/report.json')}
    assert sha(OUT/'features_train/features.pt')==prepared['aggregate_feature_sha256']
    import torch
    torch.set_num_threads(2);data=torch.load(OUT/'features_train/features.pt',map_location='cpu',weights_only=True)
    head=fit_head(data['features'],data['labels']);assert all(torch.isfinite(p).all() for p in head.parameters())
    destination.mkdir();path=destination/'last_head.pt'
    torch.save(dict(state_dict=head.state_dict(),input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],
        encoder_sha256=prepared['encoder_sha256'],fixed_steps=400),path)
    assert {p:sha(Path(p)) for p in pins}==pins
    report=dict(status='complete',train_sources=192,steps=400,no_validation_training=True,checkpoint=dict(path=str(path),sha256=sha(path)),
        encoder_sha256=prepared['encoder_sha256'],runtime_fingerprint=prepared['runtime_fingerprint'],pins=pins,
        no_source_acceptance_on_inner_outer_yet=True,no_model_deployment=True,field_accuracy=False)
    save(destination/'report.json',report);print(str(report['checkpoint']),flush=True)


if __name__=='__main__':main()
