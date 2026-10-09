"""Read only newly completed cases; produce clickable image/results gallery."""
import html
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fresh_four_examples_20261004'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    progress=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    total=len(manifest.get('cases',manifest['images']))
    rows=progress['cases']
    content=[]
    labels={'disconnected':'断接样例','normal':'正常样例','damaged':'损伤样例','misrouted':'走线异常样例'}
    reasons={'local_alignment_not_supported':'主流程采用了局部对齐，但端口分支暂不支持这一坐标变换，因而安全退出；0个提示不是正常识别成功。'}
    statuses={'applied':'已执行','fallback':'安全退出','not_run':'未执行'}
    for row in rows:
        stem=Path(row['image']).stem
        case=OUT/row.get('id',stem)
        assert Path(row['comparison'])==case/'comparison.png' and (case/'result.json').is_file()
        label=labels.get(stem.split('_')[0],row.get('id','机柜样例'))
        reason=reasons.get(row.get('port_reason'),row.get('port_reason') or '')
        relative=lambda p:Path(p).relative_to(OUT).as_posix()
        text=(f"融合复核框 {row['fusion_cues']} 个；端口主要 {row['main_ports']} 个、"
              f"补充 {row['supplementary_ports']} 个。端口状态：{statuses.get(row['ports_status'],row['ports_status'])}。"
              f"耗时 {row['seconds']/60:.1f} 分钟。")
        display=case/'desktop_result_fonts_loaded.png'
        links=[('本轮完整报告',Path(row['output'])/'report.json'),
               ('桌面结果展示',display if display.is_file() else Path(row['desktop']))]
        if row.get('port_overlay'):
            links.append(('完整分辨率端口图',Path(row['port_overlay'])))
        links.append(('完整分辨率融合图',Path(row['output'])/'sam3_fusion_boxes.jpg'))
        anchors=' · '.join(f'<a href="{html.escape(relative(p),quote=True)}">{html.escape(title)}</a>'
                           for title,p in links if p.is_file())
        legend='<p>右下黄色大框是原差异区域；青色、紫色小框才是主要、补充端口提示。</p>' if row.get('port_overlay') else ''
        content.append(f'<section><h2>{label} · {html.escape(row["image"])}</h2>'
            f'<p>{html.escape(text)}</p><p>{html.escape(reason)}</p>'+legend+
            f'<a href="{html.escape(case.name)}/comparison.png"><img src="{html.escape(case.name)}/comparison.png" alt="本轮原图及检测结果"></a>'
            f'<p>{anchors}</p></section>')
    state='全部运行完成' if progress['status']=='complete' else f'状态 {progress["status"]}；已完成 {len(rows)}/{total}；当前：{progress.get("case","")}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page+='<title>本轮检测结果</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#202124}main{max-width:1180px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:19px}section{background:white;margin:25px 0;padding:20px;border:1px solid #ddd;border-radius:6px}p{line-height:1.65}img{display:block;width:100%;height:auto}a{color:#245a9b}header p{color:#555}</style><main>'
    page+=f'<header><h1>本轮检测结果</h1><p>{html.escape(state)}</p><p>原图重新配准、DINO分析、参考图及待检图SAM分割、已验收端口增强。结果仍需人工复核。</p></header>'
    page+=''.join(content)+'</main></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(status=progress['status'],completed_cases=len(rows),gallery=str(OUT/'index.html')),ensure_ascii=False))


if __name__=='__main__':
    main()
