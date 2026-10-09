"""Package verified, immutable demo evidence with non-actuating recheck actions."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from bundle_recheck_plan import plan
from core import sha256
from run_audit import source_pins
from run_review import read, save

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    'same_visible_bundle_attachment_supported': '走线改变，可见线束接法与参考一致',
    'visible_socket_attachment_change_supported': '参考插座接点露出，支持可见插接状态变化',
    'insufficient_evidence': '遮挡或证据不足，保留待复核',
}


def verified_rows(report, audit):
    if report.get('status') != 'complete' or audit.get('status') != 'PASS':
        raise ValueError('completed demo and successful replay required')
    if report.get('reference_review_confirmed') is not True:
        raise ValueError('human reference review required')
    if report.get('deployed_to_E_mainline') is not False:
        raise ValueError('delivery must preserve research-only status')
    if report.get('new_confirmed_electrical_connections') != 0 or report.get('electrical_disconnections_confirmed') != 0:
        raise ValueError('no electrical certification in this demo')
    if not audit.get('native_component_anchor_support_independently_replayed') or not audit.get('typed_decisions_independently_replayed'):
        raise ValueError('independent component/decision replay required')
    rows = []
    for case in report['cases']:
        comparison = case['comparison']
        rows.append({'id': case['id'], 'result': LABELS[comparison['decision']],
                     'comparison': comparison, 'recheck_plan': plan(comparison)})
    if len(rows) != 3 or len({row['id'] for row in rows}) != 3:
        raise ValueError('exactly three distinct demonstration cases required')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/mendeley_confirmed_bundle_delivery_20261006')
    out = parser.parse_args().output.resolve()
    source = ROOT/'artifacts/mendeley_confirmed_bundle_review_20261006'
    audit_path = ROOT/'artifacts/mendeley_confirmed_bundle_audit_20261006/report.json'
    protocol_path = ROOT/'artifacts/mendeley_confirmed_bundle_demo_20261006/protocol.json'
    report, audit, protocol = read(source/'report.json'), read(audit_path), read(protocol_path)
    for inventory in [report['pins'], audit['pins'], protocol['pins']]:
        if any(sha256(path) != digest for path, digest in inventory.items()):
            raise ValueError('source/code/evidence fingerprints changed')
    if source_pins() != protocol['mainline_pins']:
        raise ValueError('E mainline source or dirty worktree changed')
    rows = verified_rows(report, audit)
    artifacts = [source/'report.json', audit_path, protocol_path, source/'decision_matrix.png',
                 Path(__file__), Path(__file__).with_name('bundle_recheck_plan.py')]
    pins = {str(path): sha256(path) for path in artifacts}
    result = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'human_reference_once_visible_CPU_fan_multiwire_bundle', 'cases': rows,
              'fresh_original_images': report['fresh_original_images'],
              'fresh_SAM_encoders': report['fresh_SAM_encoders'],
              'fresh_SAM_decoders': report['fresh_SAM_decoders'],
              'native_masks_verified': audit['native_masks_verified'],
              'independent_replay_status': audit['status'],
              'posthoc_selected_demo_not_blind_accuracy': True,
              'electrical_correctness': 'not_assessed', 'deployed': False,
              'recheck_execution_status': 'not_performed',
              'decision_matrix': str(source/'decision_matrix.png'), 'pins': pins}
    lines = ['# Mendeley 风扇线束：完整可见关系复核演示', '',
             '2026-10-06：用户一次确认正常参考后，从 4 张原图完成本轮复核。', '',
             '## 本轮实际结果', '',
             '| 示例 | 可见证据结论 | 下一动作 |',
             '| --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['id']} | {row['result']} | {row['recheck_plan']['action']['message']} |")
    lines += ['', '## 实际运行与核查', '',
              '- 原图重新解码；3 张待检图重新做全局及局部定位，4 张图重新计算插座外观证据。',
              '- 参考、改走线、接点露出三张图分别重新运行 SAM：3 次图像编码、6 次解码，共 36 个原生遮罩。遮挡图外观不确定，停止推断。',
              '- 独立重算 4 张原图特征、36 个原生遮罩与端点像素支持，以及三种结论，报告 PASS。',
              '- 已实际查看三结果展示图和全部 6 张原始遮罩联系表。',
              '- 保留全部遮罩和连通分量，不拼接缺口；旧单根导线门槛未改变。E 主线与既有脏工作树指纹保持一致。', '',
              '## 使用边界', '',
              '这是一组事先看过的同设备素材的定性展示，不是盲测，不提供跨机柜现场准确率。',
              '新增判断只到多芯线束的可见关系：不证明每根芯线、隐藏端子、插头完全就位或电气导通。',
              '“插接变化”只支持该插座外观变化；该示例风扇出线处定位未通过，不能把它扩写为已证实风扇电气断路。',
              '输出的人工复核与补拍只是建议，尚未执行；没有自动纠正实物。当前为独立研究入口，尚未部署到 E 主线网页/窗口。', '',
              '## 证据', '',
              f"![三种原图复核结果]({(source/'decision_matrix.png').as_posix()})", '',
              f"[完整分析]({(source/'report.json').as_posix()}) · [独立复核]({audit_path.as_posix()})", '']
    if source_pins() != protocol['mainline_pins'] or any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('fingerprints changed during report construction')
    out.mkdir(parents=True, exist_ok=False)
    save(out/'report.json', result)
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    loaded = read(out/'report.json')
    if loaded != result or len(loaded['cases']) != 3:
        raise ValueError('delivery readback mismatch')
    print('PASS: 3 non-actuating review plans; immutable evidence and E pins verified')


if __name__ == '__main__':
    main()
