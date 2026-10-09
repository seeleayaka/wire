"""Fixed diagnostic intervention on ONE already failed source cue.

Not a policy, fitting set or extra model vote. Does not infer connector identity
or suppress the original cue. Compare real pair with both exact self pairs.
"""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,embeddings,paired_features

OUT=ROOT/'artifacts/allport480_normal_pair_intervention_20261005'
CASE=ROOT/'artifacts/allport480_source_20261005/train/normal_008_predictions.json'


def main():
    if OUT.exists():raise FileExistsError('preserve fixed intervention')
    import psutil
    if psutil.virtual_memory().available<6*2**30:raise RuntimeError('protect running source experiment memory')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch,cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    frozen=native_pose_runtime_fingerprint(REPO);case=load(CASE);cue=case['trial']['all_predictions']
    if case['current']['all_predictions'] or len(cue)!=1:raise ValueError('diagnostic target changed')
    cue=cue[0];source=DATA/'images/train01'/case['image'];reference=DATA/'images/train01/normal_073.JPG'
    pins={str(p):sha(p) for p in (Path(__file__),CASE,source,reference,REPO/HEAD_RELATIVE,REPO/'inspection_agent/paired_port_features.py')}
    if pins[str(source)]!=case['source_sha256'] or pins[str(reference)]!=REFERENCE_SHA or pins[str(REPO/HEAD_RELATIVE)]!=HEAD_SHA:
        raise ValueError('fixed actual input/head identity mismatch')
    observed=read_image(source);ref=read_image(reference)
    expected,_=expected_in_source(ref,case['alignment']['source_to_reference_homography'],observed.shape[:2])
    started=time.monotonic()
    import dino_feature_diff as dino
    encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(1)
    head=torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    box=[cue['box_xyxy']];obs=embeddings(encoder,observed,box);exp=embeddings(encoder,expected,box)
    vectors=paired_features(torch.cat([obs,obs,exp]),torch.cat([exp,obs,exp]))
    with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
    if abs(scores[0][cue['class_id']+1]-cue['paired_semantic_probability'])>1e-5:raise ValueError('actual-pair head replay differs')
    if any(sha(p)!=v for p,v in pins.items()) or native_pose_runtime_fingerprint(REPO)!=frozen:raise ValueError('diagnostic input/runtime drift')
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',seconds=round(time.monotonic()-started,2),pins=pins,runtime=frozen,
        interventions=[dict(pair=name,probabilities=score) for name,score in zip(['actual_observed_expected','observed_exact_self','expected_exact_self'],scores)],
        original_cue=cue,classes=['other','unplugged_plug','unplugged_jack'],independent_model_observers=1,
        diagnostic_one_failure_not_a_tuning_set=True,actual_pair_replay_tolerance=1e-5,
        same_physical_connector_confirmed=False,no_fitting=True,no_policy_change=True,no_threshold_change=True,
        no_cue_suppression=True,no_deployment=True,field_accuracy=None))
    print(dict(status='complete',scores=scores,seconds=round(time.monotonic()-started,2)),flush=True)


if __name__=='__main__':main()
