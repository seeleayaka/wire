"""Actual desktop workers from original pixels, isolated fresh caches per case."""
import copy
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/fresh_four_examples_20261004'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',
    YOLO_AUTOINSTALL='False',PYTHONPROJECT10_DINO_CACHE=str(OUT/'bootstrap_dino_cache'))
sys.path[:0]=[str(REPO),str(REPO/'prototype')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(8):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==7:
                raise
            time.sleep(.05)


def montage(case,reference,source,output,port_overlay):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    panels=[('参考原图',reference),('待检原图',source),
            ('DINO + SAM 最终复核框',output/'sam3_fusion_boxes.jpg'),
            ('已验收端口增强',port_overlay)]
    canvas=Image.new('RGB',(1500,1240),'#f4f5f7')
    draw=ImageDraw.Draw(canvas)
    for index,(title,path) in enumerate(panels):
        x=(index%2)*750;y=(index//2)*620
        draw.text((x+18,y+10),title,font=font,fill='#222222')
        if path is not None and Path(path).is_file():
            with Image.open(path) as opened:
                image=opened.convert('RGB')
                image.thumbnail((714,555))
            canvas.paste(image,(x+18+(714-image.width)//2,y+50+(555-image.height)//2))
        else:
            draw.text((x+25,y+100),'本阶段未产生叠图，见结果说明',font=font,fill='#666666')
    canvas.save(case/'comparison.png')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--suite',type=Path)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if OUT.exists():
        raise FileExistsError('fresh output required')
    OUT.mkdir()
    import psutil
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    import dino_feature_diff
    from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
    from inspection_agent.optional_port_crop_review import SCENE
    from inspection_agent import InspectionTask
    reference=DATA/'images/train01/normal_073.JPG'
    paths=[sorted((DATA/'images/test01').glob(kind+'_*.JPG'))[0]
           for kind in ('disconnected','normal','damaged','misrouted')]
    cases=[dict(id=p.stem,reference=str(reference),inspection=str(p),
                rois=[[.03,.04,.97,.96]],ports=True) for p in paths]
    if args.suite:
        cases=json.loads(args.suite.read_text(encoding='utf-8'))['cases']
        if not cases or len({c['id'] for c in cases})!=len(cases):
            raise ValueError('nonempty unique cases required')
        for c in cases:
            if Path(c['id']).name!=c['id'] or c['id'] in ('.','..'):
                raise ValueError('unsafe case id')
        paths=[Path(c['inspection']) for c in cases]
    source_files=[Path(__file__),REPO/'prototype/assembly_auto_review_dino.py',
        REPO/'prototype/sam3_wire_fusion.py',REPO/'prototype/dino_feature_diff.py',
        REPO/'inspection_agent/port_rescue_gui.py',REPO/'inspection_agent/paired_native_pose.py']
    pins={str(p):sha(p) for p in source_files+[Path(c['reference']) for c in cases]+paths}
    runtime=native_pose_runtime_fingerprint(REPO)
    settings=gui.sam3_wire_fusion.load_settings()
    gui.sam3_wire_fusion.validate_resources(settings)
    pins[str(settings.checkpoint)]=sha(settings.checkpoint)
    manifest=dict(reference=str(reference),selection='first alphabetic per category, disconnected/normal/damaged/misrouted',
        images=[str(p) for p in paths],pins=pins,runtime=runtime,
        new_alignment=True,new_DINO=True,new_reference_SAM_each_case=True,new_inspection_SAM=True,
        prior_intermediate_artifacts_read=False,optional_accepted_ports=True,
        unfinished_training_weights_used=False,roi=[[.03,.04,.97,.96]],human_confirmation=False)
    manifest['cases']=cases
    manifest['selection']='frozen external suite' if args.suite else manifest['selection']
    if args.suite:
        manifest['suite_sha256']=sha(args.suite)
    save(OUT/'manifest.json',manifest)
    app=gui.QApplication.instance() or gui.QApplication([])
    app.setStyle('Fusion')
    original_out=gui.adaptive.robust.auto.base.OUT
    original_loader=gui.sam3_wire_fusion.load_settings
    original_probe=gui.sam3_wire_fusion._run_probe
    current=dict(status='running',pid=os.getpid(),case=None,phase='starting',cases=[])
    started=time.monotonic()
    def progress(phase):
        current.update(phase=phase,seconds=round(time.monotonic()-started,2))
        save(OUT/'progress.json',current)
        print(json.dumps(dict(case=current['case'],phase=phase,seconds=current['seconds']),ensure_ascii=False),flush=True)
    def fresh_probe(source_path,output_dir,state_cache,configured):
        assert not (output_dir/'report.json').exists() and not state_cache.exists(),'SAM intermediate reuse prohibited'
        progress('fresh_SAM_reference' if Path(source_path)==reference else 'fresh_SAM_inspection')
        report,hit=original_probe(source_path,output_dir,state_cache,configured)
        assert hit is False
        return report,hit
    gui.sam3_wire_fusion._run_probe=fresh_probe
    save(OUT/'progress.json',current)
    try:
        for spec in cases:
            path=Path(spec['inspection'])
            reference=Path(spec['reference'])
            case=OUT/spec['id']
            case.mkdir()
            current['case']=path.name
            wait_deadline=time.monotonic()+7200
            while psutil.virtual_memory().available<6*2**30:
                if time.monotonic()>wait_deadline:
                    raise TimeoutError('memory wait deadline')
                progress('waiting_for_memory')
                time.sleep(15)
            begin=time.monotonic()
            configured=replace(settings,cache_dir=case/'fresh_sam_cache',threads=2)
            assert not configured.cache_dir.exists()
            gui.sam3_wire_fusion.load_settings=lambda *a,**k:configured
            dino_feature_diff.CACHE_DIR=case/'fresh_dino_cache'
            assert not dino_feature_diff.CACHE_DIR.exists()
            gui.adaptive.robust.auto.base.OUT=case/'desktop_output'
            gui.adaptive.robust.auto.base.OUT.mkdir()
            window=gui.DINOReview()
            window.resize(1400,950)
            window.reference.setText(str(reference))
            window.inspection.setText(str(path))
            window.recipe=dict(reference_image=str(reference),check_rois=copy.deepcopy(spec['rois']))
            window.port_scene_combo.setCurrentIndex(window.port_scene_combo.findData(SCENE))
            assert window.port_scene_combo.currentData()==SCENE
            switches=(window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch)
            assert not any(c.isChecked() for c in switches)
            cv2.setRNGSeed(0)
            progress('actual_detect_button_alignment_DINO')
            window.run_button.click()
            assert window.initial_worker is not None
            window.initial_worker.stage_changed.connect(progress)
            deadline=time.monotonic()+7200
            while window.initial_worker is not None or window.sam3_worker is not None:
                app.processEvents()
                if time.monotonic()>deadline:
                    raise TimeoutError('desktop pipeline deadline; no success claimed')
                time.sleep(.05)
            app.processEvents()
            if window.current_output is None:
                raise RuntimeError('desktop analysis produced no output')
            output=Path(window.current_output)
            report_path=output/'report.json'
            final=json.loads(report_path.read_text(encoding='utf-8'))
            save(case/'fresh_default_report.json',final)
            sam=final.get('sam3_fusion',{})
            if sam.get('status')=='ok':
                assert not sam['reference_sam3']['cache_hit'] and not sam['inspection_sam3']['cache_hit']
            elif final.get('decision')!='alignment_uncertain_manual_review':
                raise RuntimeError('fresh SAM failed: '+str(sam.get('error')))
            port_overlay=None
            if sam.get('status')=='ok' and spec.get('ports',False):
                for switch in switches:
                    switch.setChecked(True)
                window._update_port_controls()
                if window.rescue_button.isEnabled():
                    progress('fresh_accepted_port_models_and_reference_checks')
                    window.rescue_button.click()
                    assert window.rescue_worker is not None
                    port_deadline=time.monotonic()+1800
                    while window.rescue_worker is not None:
                        app.processEvents()
                        if time.monotonic()>port_deadline:
                            raise TimeoutError('port worker deadline')
                        time.sleep(.05)
                    app.processEvents()
            final=json.loads(report_path.read_text(encoding='utf-8'))
            ports=final.get('independent_port_rescue',{})
            if ports.get('evidence_path'):
                candidate=Path(ports['evidence_path'])/'overlay.jpg'
                port_overlay=candidate if candidate.is_file() else None
            assert window.agent_task_path and window.agent_task_path.is_file()
            task=InspectionTask.load(window.agent_task_path).to_report()
            assert not task['human_conclusions'] and task['state']=='awaiting_human_review'
            app.processEvents()
            window.grab().save(str(case/'desktop_result.png'))
            montage(case,reference,path,output,port_overlay)
            row=dict(id=spec['id'],case_directory=str(case),reference=str(reference),
                source=str(path),scene=spec.get('scene','mendeley'),
                image=path.name,output=str(output),comparison=str(case/'comparison.png'),
                desktop=str(case/'desktop_result.png'),decision=final.get('decision'),
                sam_status=sam.get('status'),fresh_reference_SAM=sam.get('status')=='ok' and not sam['reference_sam3']['cache_hit'],
                fresh_inspection_SAM=sam.get('status')=='ok' and not sam['inspection_sam3']['cache_hit'],
                dino_cues=len(final.get('dino_review_regions',[])),fusion_cues=len(final.get('review_regions',[])),
                ports_status=ports.get('status','not_run'),port_reason=ports.get('fallback_reason'),
                main_ports=len(ports.get('rescue_hints',[])),supplementary_ports=len(ports.get('supplementary_hints',[])),
                port_overlay=str(port_overlay) if port_overlay else None,task=str(window.agent_task_path),
                seconds=round(time.monotonic()-begin,2),human_confirmation=False)
            save(case/'result.json',row)
            current['cases'].append(row)
            window.close()
            app.processEvents()
            assert all(sha(Path(p))==v for p,v in pins.items())
            assert native_pose_runtime_fingerprint(REPO)==runtime
            progress('case_complete')
        current.update(status='complete',phase='complete',seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',current)
        save(OUT/'progress.json',current)
    except BaseException as error:
        current.update(status='failed',error=type(error).__name__+': '+str(error))
        save(OUT/'progress.json',current)
        raise
    finally:
        gui.adaptive.robust.auto.base.OUT=original_out
        gui.sam3_wire_fusion.load_settings=original_loader
        gui.sam3_wire_fusion._run_probe=original_probe


if __name__=='__main__':
    main()
