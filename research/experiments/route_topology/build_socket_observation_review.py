"""Independent replay audit and read-only, offline observation display."""
import html
import json
from pathlib import Path
from run_prompt_contrast import digest, save, verify
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/socket_observation_layer_20261008'


def main():
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    verify(report['pins'])
    assert source_pins() == report['mainline_pins']
    old = json.loads((ROOT / 'artifacts/endpoint_pair_expanded30_v3_20261008/report.json').read_text(encoding='utf-8'))
    photo_dir = ROOT / 'artifacts/wire_identity_photo_review_20261008'
    manifest = json.loads((photo_dir / 'manifest.json').read_text(encoding='utf-8'))
    photos = {r['case_id']: r for r in manifest['samples']}
    assert len(report['cases']) == len(old['cases']) == len(photos) == 30
    cards = []
    exposure = []
    conflicts = []
    pins = {str(OUT / 'report.json'): digest(OUT / 'report.json'),
            str(photo_dir / 'manifest.json'): digest(photo_dir / 'manifest.json')}
    names = {'socket_contacts_exposed_observed': '插座接点露出（局部观察）',
             'mating_housing_observed': '配对插头外壳可见（局部观察）',
             'unknown': '插座状态无法确认'}
    for previous, current in zip(old['cases'], report['cases']):
        assert {k: v for k, v in current.items() if k != 'socket_observation'} == previous
        photo = photos[current['id']]
        assert digest(current['source_path']) == current['source_binding']['image_sha256'] == photo['source_sha256']
        # Derive independently from the original record, not by calling the layer.
        expected = 'unknown'
        if previous['local_anchor_support']['FAN_CPU'] is True:
            if previous['socket_phenotype_observed'] == 'socket_contacts_exposed':
                expected = 'socket_contacts_exposed_observed'
            elif previous['socket_phenotype_observed'] == 'mating_body_visible':
                expected = 'mating_housing_observed'
        observed = current['socket_observation']
        assert observed['state'] == expected
        assert observed['automatic_fault_confirmed'] is False
        assert observed['electrical_continuity'] == 'not_assessed'
        if expected == 'socket_contacts_exposed_observed':
            exposure.append(current['id'])
            if previous['socket_evidence_conflict']:
                conflicts.append(current['id'])
        details = photo_dir / photo['detail_image']
        overview = photo_dir / photo['overview_image']
        assert details.is_file() and overview.is_file()
        pins.update({str(details): digest(details), str(overview): digest(overview)})
        conflict_text = ('插座分类与线束掩膜存在冲突，需要人工复核。'
                         if previous['socket_evidence_conflict'] else '未改变原连接判定。')
        cards.append(f'''<section data-case="{html.escape(current['id'])}">
<h2>{html.escape(current['id'])} · {names[expected]}</h2>
<p>{conflict_text}</p><div class="photos">
<img loading="lazy" src="../{photo_dir.name}/{html.escape(details.name)}" alt="原图局部">
<img loading="lazy" src="../{photo_dir.name}/{html.escape(overview.name)}" alt="原图全景"></div>
<p class="muted">原连接结果：{html.escape(previous['decision'])}<br>
原因：{html.escape(previous['reason'])}</p></section>''')
    assert exposure == ['case_06', 'case_07', 'case_09', 'case_10']
    assert conflicts == ['case_07', 'case_09']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>插座观察与连接复核</title><style>
body{background:#f5f6f7;color:#25313a;font:16px system-ui;margin:0}
main{max-width:1080px;margin:24px auto;padding:0 16px}
header,section{background:white;border:1px solid #dce1e5;border-radius:8px;padding:20px;margin:16px 0}
h1{font-size:24px}h2{font-size:18px}p{line-height:1.6;overflow-wrap:anywhere}.muted{color:#66717a;font-size:14px}
.photos{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}img{width:100%;max-height:420px;object-fit:contain}
@media(max-width:700px){.photos{grid-template-columns:1fr}}</style><main><header>
<h1>插座观察与连接复核</h1><p>局部插座观察与整条连接结论分别显示。接点露出不等于已经证实断路；线束候选不等于已经证实电气连接正确。</p>
<p class="muted">30张既有照片结果重放；复用了原分类与SAM，不是重新推理，也不是新准确率。第7、9张保留冲突提示。第8张仍无法确认。未接入正式程序，无上传或自动修改。</p>
</header>''' + ''.join(cards) + '</main></html>'
    assert '<script' not in page and 'https://' not in page
    (OUT / 'review.html').write_text(page, encoding='utf-8')
    verify(pins)
    assert source_pins() == report['mainline_pins']
    save(OUT / 'audit_report.json', dict(status='PASS', cases_checked=30,
         original_fields_unchanged=True, original_source_hashes_verified=True,
         independent_observation_derivation=True, exposure_ids=exposure,
         conflict_ids=conflicts, confirmed_faults=0, new_classifications=0,
         photo_files_verified=60, html_cards=30, browser_render_verified=False,
         mainline_unchanged=True, pins=pins))
    print(json.dumps(dict(status='PASS', cases=30, conflicts=conflicts,
                         new_classifications=0, browser_render_verified=False)))


if __name__ == '__main__':
    main()
