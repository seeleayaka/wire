"""Fixed source-group OOF conditional crop classifier, not inspection accuracy."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,load,save,sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from port_semantic_verifier import fit_head,PROBABILITY_GATE,MIN_PRECISION,MIN_RECALL


def main():
    destination=OUT/'heads_oof'
    if destination.exists():raise FileExistsError('Preserve OOF heads')
    source=OUT/'features_train';report=load(source/'report.json');assert report['status']=='complete' and len(report['source_groups'])==192
    assert report['gt_targets']==344 and report['no_validation_training'] and report['frozen_encoder_unchanged']
    assert {p:sha(Path(p)) for p in report['pins']}==report['pins'] and {p:sha(Path(p)) for p in report['cache_pins']}==report['cache_pins']
    assert resolution_runtime_fingerprint(REPO)==report['runtime_fingerprint']
    assert sha(source/'features.pt')==report['aggregate_feature_sha256'] and sha(source/'samples.json')==report['samples_sha256']
    destination.mkdir();pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_verifier.py'),source/'report.json',source/'features.pt',source/'samples.json')}
    save(destination/'protocol.json',dict(pins=pins,probability_gate=PROBABILITY_GATE,min_precision=MIN_PRECISION,min_recall=MIN_RECALL,
        fixed400_linear_AdamW_steps=True,whole_source_classifier_folds=True,YOLO_proposals_not_independent_OOF=True,
        GT_crop_recall_denominator_includes_abstained_GT=True,negative_reference_and_actual_proposal_errors_in_precision=True,
        no_validation_training=True,no_best_checkpoint_or_score_tuning=True,no_deployment=True,field_accuracy=False))
    import torch
    torch.set_num_threads(2);data=torch.load(source/'features.pt',map_location='cpu',weights_only=True);metadata=load(source/'samples.json')
    features,labels,folds=data['features'],data['labels'],data['folds'];assert features.shape==(len(metadata),6144)
    assert all(r['label']==int(labels[i]) and r['fold']==int(folds[i]) for i,r in enumerate(metadata))
    probabilities=torch.zeros((len(labels),3));head_pins={};fold_reports=[]
    for fold in range(3):
        mask=folds==fold;held={r['image'] for r in metadata if r['fold']==fold};training={r['image'] for r in metadata if r['fold']!=fold};assert held.isdisjoint(training)
        head=fit_head(features[~mask],labels[~mask]);assert all(torch.isfinite(p).all() for p in head.parameters())
        with torch.inference_mode():probabilities[mask]=head(features[mask]).softmax(dim=1)
        path=destination/f'fold{fold}_head.pt';torch.save(head.state_dict(),path);head_pins[str(path)]=sha(path)
        fold_reports.append(dict(fold=fold,training_source_count=len(training),held_source_count=len(held),held_samples=int(mask.sum())))
        save(destination/'progress.json',dict(status='running',completed_folds=fold+1,total=3))
    scores,classes=probabilities.max(dim=1);accepted=(scores>=PROBABILITY_GATE)&(classes>0);correct=accepted&(classes==labels)
    gt_mask=torch.tensor([r['kind']=='gt_port' for r in metadata]);gt_correct=int((correct&gt_mask).sum())
    tp=int(correct.sum());unmatched=int((accepted&~correct).sum());precision=tp/max(1,tp+unmatched);recall=gt_correct/report['gt_targets']
    qualifies=precision>=MIN_PRECISION and recall>=MIN_RECALL
    probabilitypath=destination/'oof_probabilities.pt';torch.save(probabilities,probabilitypath)
    assert {p:sha(Path(p)) for p in pins}==pins
    result=dict(status='complete',qualifies_crop_feasibility=qualifies,summary=dict(classifier_sample_tp=tp,classifier_sample_unmatched=unmatched,
        classifier_sample_precision=precision,GT_crop_correct=gt_correct,GT_crop_targets=report['gt_targets'],GT_crop_recall=recall),
        folds=fold_reports,pins=pins,head_pins=head_pins,probabilities_sha256=sha(probabilitypath),classifier_OOF_only=True,
        YOLO_proposals_were_trained_on_all192=True,inspection_accuracy_not_evaluated=True,field_accuracy=False,no_deployment=True)
    save(destination/'report.json',result);save(destination/'progress.json',dict(status='complete',qualifies_crop_feasibility=qualifies,summary=result['summary']))
    print(str(result['summary']),flush=True)


if __name__=='__main__':main()
