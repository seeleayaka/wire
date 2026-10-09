"""Independent scalar/graph checks. Not an independent image or model observer."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import label
from PIL import Image
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify
from run_reference_color_paths_source import exact_regions

ROOT=Path(__file__).resolve().parents[2]


def graph_path(raw,edges,ends,regions):
    labels,n=label(raw,np.ones((3,3),int));adj={i:set() for i in range(1,n+1)}
    endpoint={e['endpoint_id']:e for e in ends}
    for edge in edges:
        if edge['state'] not in ['fragment_pair_candidate','occluder_supported_fragment_candidate']:continue
        components=[]
        for k in edge['endpoint_ids']:
            x,y=np.rint(endpoint[k]['point_xy']).astype(int);c=int(labels[y,x]);assert c>0;components.append(c)
        a,b=components;adj[a].add(b);adj[b].add(a)
    visited=set()
    for start in adj:
        if start in visited:continue
        stack=[start];group=set()
        while stack:
            c=stack.pop()
            if c in group:continue
            group.add(c);stack.extend(adj[c]-group)
        visited|=group;pixels=np.isin(labels,list(group))
        if all(int((pixels&r).sum())>=8 for r in regions.values()):return True
    return False


def independent_RGB_selection(rgb,native,bins):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    eligible=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)&np.isin(hsv[:,:,0].astype(int)//10,bins)
    labels,n=label(eligible,np.ones((3,3),int));chosen=[]
    for c in range(1,n+1):
        if int(((labels==c)&native).sum())>=24:chosen.append(c)
    return np.isin(labels,chosen)


def main():
    bp=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((bp/'preparation_report.json').read_text(encoding='utf-8'));poses={c['id']:c for c in prep['cases']}
    cp=json.loads((bp/'protocol.json').read_text(encoding='utf-8'));scope=json.loads(Path(cp['confirmed_scope_path']).read_text(encoding='utf-8'))
    old=json.loads((ROOT/'artifacts/soft_tangent_source_20261008/report.json').read_text(encoding='utf-8'))
    originals={r['id']:r for r in old['cases']};summary=[]
    for name,kind in [('occluder_tangent_source_20261008','families'),('occluder_native_bundle_20261008','native'),('seeded_visible_occlusion_20261008','RGB')]:
        directory=ROOT/'artifacts'/name;p=json.loads((directory/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);assert source_pins()==p['mainline_pins']
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'));checks=0;recovered=0;accepted=0;retained=0
        for row in report['cases']:
            binding=row['source_binding'];assert digest(binding['path'])==binding['image_sha256']
            rgb=np.asarray(Image.open(binding['path']).convert('RGB').crop(row['crop_box_xyxy']))
            regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors'])
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10
            colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32);record_paths=[]
            for rec in row['records']:
                assert digest(rec['mask_path'])==rec['mask_sha256'];native=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128
                original=next(r for r in originals[row['id']]['records'] if r['record_id']==rec['record_id'])
                if kind=='native':
                    assert graph_path(native,[],rec['endpoints'],regions)==bool(rec['old_native_paths'])
                    actual=graph_path(native,rec['edges'],rec['endpoints'],regions);assert actual==bool(rec['new_paths']);record_paths.append(actual);checks+=2
                    families=[rec]
                else:
                    selected=native
                    if kind=='RGB':
                        selected=independent_RGB_selection(rgb,native,sorted({b for f in rec['families'] for b in f['hue_bins']}))
                        output=np.asarray(Image.open(directory/(row['id']+'_'+rec['record_id']+'_RGB_supplement.png')).convert('L'))>=128
                        assert np.array_equal(selected,output);count=int((selected&~native).sum());assert count==rec['supplement']['recovered_RGB_pixels'];recovered+=count;checks+=2
                    family_paths=[];families=rec['families']
                    for i,family in enumerate(families):
                        raw=colored&selected&np.isin(bins,family['hue_bins']);actual=graph_path(raw,family['edges'],family['endpoints'],regions)
                        if kind=='RGB':
                            assert actual==bool(family['supplemental_paths']);actual=actual or bool(original['families'][i]['anchor_path_candidates'])
                        else:
                            assert actual==bool(family['anchor_path_candidates'])
                            oldedges=original['families'][i]['edges'];assert len(oldedges)==len(family['edges'])
                            assert family['endpoints']==original['families'][i]['endpoints']
                            for a,b in zip(oldedges,family['edges']):
                                assert a['score']==b['score'] and a['virtual_curve_xy']==b['virtual_curve_xy'] and a['endpoint_ids']==b['endpoint_ids']
                                if a['state']=='fragment_pair_candidate':assert b['state']==a['state'];retained+=1
                        family_paths.append(actual);checks+=1
                    record_paths.append(all(family_paths))
                for family in families:
                    for edge in family['edges']:
                        if edge['state']=='occluder_supported_fragment_candidate':
                            assert edge['score']<=.5 and edge['occluder_witness']['candidate'] and not edge['physical_identity_confirmed'] and edge['observed_gap_pixels']==0;accepted+=1
            assert any(record_paths)==row['new_path'];checks+=1
        assert report['gains']==[] and report['losses']==[] and report['exposed_source_candidates']==[] and not report['source_gate_passed']
        if kind=='RGB':assert recovered==report['recovered_RGB_pixels']==10974
        summary.append(dict(experiment=name,cases=len(report['cases']),checks=checks,added_fragment_candidates=accepted,old_edges_preserved=retained,recovered_RGB_pixels=recovered,status='PASS'))
    result=dict(status='PASS',experiments=summary,shared_HSV_conversion_and_registered_regions=True,
        independent_Scipy_components_and_graph_search=True,independent_new_model_observer=False,
        geometry_fit_and_witness_not_independently_reimplemented=True,new_confirmed_connections=0,mainline_unchanged=True)
    save(ROOT/'artifacts/occlusion_recovery_audit_20261008.json',result);print(json.dumps(result))


if __name__=='__main__':main()
