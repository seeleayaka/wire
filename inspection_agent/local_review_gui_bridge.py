"""Current-view gate for the GUI; no test bypass or fault confirmation."""
import hashlib
import json
from pathlib import Path
from .workflow import InspectionTask
from .terminal_mapping import image_binding
from .local_evidence_bridge import attach_review_file

def task_fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def import_for_current_view(task_path,packet_path,*,reference,inspection,expected_task_sha):
    target=Path(task_path)
    if task_fingerprint(target)!=expected_task_sha:raise ValueError('工单已变化，请重新检查当前工单后导入')
    task=InspectionTask.load(target);report=task.to_report()
    if task.state!='awaiting_human_review':raise ValueError('当前工单不是等待人工复核状态')
    for name,value in [('reference',reference),('inspection',inspection)]:
        if Path(value).resolve()!=Path(report['inputs'][name]).resolve():raise ValueError('当前图片与工单不一致，请先完成当前图片的视觉分析')
    source=Path(packet_path);packet=json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(packet,dict) or packet.get('schema_version')!=1 or set(packet)!={'schema_version','bundle','review','evidence_files','plan'}:raise ValueError('导入清单格式错误')
    def resolve(key,optional=False):
        value=packet[key]
        if optional and value is None:return None
        if not isinstance(value,str) or not value.strip():raise ValueError('导入清单缺少文件路径')
        return (source.parent/Path(value)).resolve()
    bp,rp,mp,pp=resolve('bundle'),resolve('review'),resolve('evidence_files'),resolve('plan',True)
    bundle=json.loads(bp.read_text(encoding='utf-8'));binding=image_binding(Path(inspection))
    matches=[case for case,version in bundle['source_versions'].items() if version['image_binding']==binding]
    if len(matches)!=1:raise ValueError('证据包没有唯一匹配当前照片，不能自动选择视角')
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    files={case:{k:(mp.parent/Path(v)).resolve() for k,v in paths.items()} for case,paths in manifest.items()}
    if task_fingerprint(target)!=expected_task_sha:raise ValueError('选择文件期间工单已变化，导入取消')
    result=attach_review_file(target,bp,rp,files,matches[0],plan_path=pp,allow_test_records=False)
    counts=result['candidate_summary'];plan_text='未确认预期草稿已附加' if result['expected_plan_draft'] else '未附加预期草稿'
    summary=('已附加局部意见：支持{supported} / 排除{rejected} / 无法确认{uncertain} / 未处理{pending}。'.format(**{s:counts.get(s,0) for s in ('supported','rejected','uncertain','pending')})
             +'\n'+plan_text+'；工单仍待人工复核，未确认故障、未比较拓扑。')
    return result,summary
