"""Build a source-verified read-only summary from an exported review."""
import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path
from review_readiness import summarize_review

ROOT=Path(__file__).resolve().parents[1]
LABELS={'conflicting_support':'同一线段跨入口支持，或同入口支持多个候选',
 'no_visible_candidate':'没有可见候选；不能据此判缺线','pending_candidates':'仍有候选未复核',
 'uncertain_evidence':'存在无法确认的证据','all_candidates_rejected_not_missing_wire':'候选全部排除；仍不能判缺线',
 'local_support_not_connection':'仅支持局部关系，不是连接边','port_identity_unconfirmed':'入口范围与端子身份未确认',
 'no_verified_complete_cable_observations':'缺少已核验的完整线缆两端与身份观察',
 'test_record_not_operator_evidence':'当前是测试记录，不是真实操作确认',
 'missing_port_map':'缺少端子地图','expected_scope_unconfirmed':'有限预期范围未确认',
 'expected_connections_unknown':'预期连接表未知','expected_table_unconfirmed':'预期表及依据未确认',
 'conflicting_local_support':'局部支持存在冲突'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('review',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    # Reuse the existing live hash verifier. Failure produces no summary artifacts.
    subprocess.run([sys.executable,str(ROOT/'experiments/validate_local_review_file.py'),str(args.review)],check=True)
    bundle=json.loads((ROOT/'artifacts/local_review_ui_v2_20261001/bundle.json').read_text(encoding='utf-8'))
    review=json.loads(args.review.read_text(encoding='utf-8'))
    maps={case:json.loads((ROOT/f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json').read_text(encoding='utf-8')) for case in bundle['source_versions']}
    report=summarize_review(bundle,review,maps)
    report['review_file_sha256']=hashlib.sha256(args.review.read_bytes()).hexdigest()
    e=html.escape
    sections=[]
    for gate in report['topology_readiness']:
        sections.append('<section><h2>'+e(gate['case'])+'：证据不足，未执行拓扑比较</h2><ul>'+''.join('<li>'+e(LABELS[b])+'</li>' for b in gate['blockers'])+'</ul></section>')
    for row in report['entries']:
        opinions=''.join('<li>'+e(c['record_id'])+' / '+e(c['state'])+'：'+e(c['evidence_note'] or '尚未填写')+'</li>' for c in row['candidate_reviews'])
        sections.append('<section><h2>'+e(row['title'])+'</h2><p>'+e('；'.join(LABELS[r] for r in row['reasons']))+'</p><ul>'+opinions+'</ul><p>入口备注：'+e(row['entry_review']['evidence_note'] or '尚未填写')+'</p></section>')
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WireMind 复核汇总</title><style>body{font-family:system-ui;margin:24px auto;padding:0 16px;max-width:960px;background:#edf2f6;color:#183042}section{background:white;padding:18px;margin:16px 0;border-radius:10px}li,p{overflow-wrap:anywhere;line-height:1.6}h2{font-size:20px}</style><h1>WireMind · 复核汇总与拓扑检查入口</h1><p>记录类型：'+e(report['review_mode'])+'；署名：'+e(report['reviewer'])+'</p><p>连接边：0。没有正常、缺线或错接结论。署名与文件校验不证明意见正确。</p>'+''.join(sections)+'</html>'
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'summary.html').write_text(document,encoding='utf-8')
    print('Read-only summary written; no topology comparison or connection edges.')
if __name__=='__main__':main()
