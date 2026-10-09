"""Offline review form: user exports self-reported observations, not physical GT."""
import json
from pathlib import Path
from run_prompt_contrast import save,digest,verify

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wire_identity_photo_review_20261008'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    draft=json.loads((OUT/'ai_visual_draft.json').read_text(encoding='utf-8'))
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'));verify(report['pins'])
    labels={r['review_id']:r for r in draft['observations']}
    packet=[dict(s,AI_socket=labels[s['review_id']]['socket']) for s in manifest['samples']]
    data=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>线束照片复核</title>
<style>body{margin:0;background:#f5f6f7;color:#25313a;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:28px auto;padding:0 20px}h1{font-size:24px;font-weight:600}p{line-height:1.6}header,section{background:white;border:1px solid #dce1e5;border-radius:8px;padding:20px;margin:16px 0}.muted{color:#66717a;font-size:14px}.photos{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{display:block;max-width:100%;max-height:540px;margin:auto;object-fit:contain}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select,input{font:inherit;padding:9px 12px;border:1px solid #c7d0d8;border-radius:5px;background:white}button{cursor:pointer}button:hover{background:#edf1f4}label{display:block;margin:14px 0}select{max-width:100%}textarea{display:block;width:100%;box-sizing:border-box;min-height:72px;font:inherit;padding:10px;border:1px solid #c7d0d8;border-radius:5px}@media(max-width:700px){.photos{grid-template-columns:1fr}}</style>
<main><header><h1>线束照片复核</h1><p>请根据原图记录插座的可见状态。看不清就保留“无法确认”。这里的确认不代表电气连通，也不会修改原数据集或模型。</p><p class="muted">30张既有照片；没有SAM叠加。AI描述仅供对照，不是正确答案。导出内容需核验后才能作为人工观察记录。</p></header>
<section><div class="controls"><button id="prev">上一张</button><select id="choose" aria-label="选择照片"></select><button id="next">下一张</button><button id="overview">切换全图 / 局部</button></div><p id="caption"></p><div class="photos"><div><p class="muted">参考图：此前确认的可见线束与FAN_CPU插座</p><img src="reference_detail.png" alt="参考图"></div><div><p class="muted">当前原图</p><img id="photo" alt="当前待复核照片"></div></div><p id="AI" class="muted"></p></section>
<section><label>你的观察 <select id="socket"><option value="unknown">无法确认 / 尚未复核</option><option value="mating_housing_apparent">看得到配对插头的外壳</option><option value="contacts_exposed_apparent">插座接点露出，看不到配对插头</option></select></label><label>补充说明<textarea id="note" placeholder="例如：标签挡住端点，或需要查看全图"></textarea></label><div class="controls"><input id="reviewer" aria-label="复核者名字" placeholder="复核者名字（导出时填写）"><button id="download">导出复核记录</button></div><p id="status" role="status" class="muted">记录只保留在当前页面；关闭前请导出。不会上传照片。</p></section></main>
<script>
const packet=__PACKET__;
const names={unknown:'无法确认',mating_housing_apparent:'看得到配对插头外壳',contacts_exposed_apparent:'插座接点露出'};
const results=packet.map(s=>({review_id:s.review_id,case_id:s.case_id,source_sha256:s.source_sha256,socket:'unknown',note:''}));
let index=0,whole=false;
const el=id=>document.getElementById(id);
packet.forEach((s,i)=>{const option=document.createElement('option');option.value=i;option.textContent=s.review_id+' / '+s.case_id;el('choose').appendChild(option)});
function saveCurrent(){results[index].socket=el('socket').value;results[index].note=el('note').value}
function show(){const s=packet[index];el('choose').value=index;el('photo').src=whole?s.overview_image:s.detail_image;el('caption').textContent=s.review_id+' / '+s.case_id+' · '+(index+1)+' / 30';el('AI').textContent='AI复核草稿：'+names[s.AI_socket]+'；未经人工确认。';el('socket').value=results[index].socket;el('note').value=results[index].note;el('prev').disabled=index===0;el('next').disabled=index===packet.length-1}
el('prev').onclick=()=>{saveCurrent();index=Math.max(0,index-1);show()};el('next').onclick=()=>{saveCurrent();index=Math.min(packet.length-1,index+1);show()};el('choose').onchange=()=>{saveCurrent();index=Number(el('choose').value);show()};el('overview').onclick=()=>{whole=!whole;show()};
el('download').onclick=()=>{saveCurrent();const name=el('reviewer').value.trim();if(!name){el('status').textContent='请填写复核者名字后再导出。';return}const record={schema_version:1,reviewer_type:'human_self_reported',reviewer_name:name,created_at:new Date().toISOString(),human_confirmed_by_agent:false,independent_physical_ground_truth:false,electrical_continuity:'not_assessed',cross_photo_same_physical_wire_identity:'unknown',requires_review_import_verification:true,observations:results};const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='线束照片_人工观察记录.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);el('status').textContent='已导出自报观察记录。原数据、模型和正式结论没有变化。'};show();
</script></html>'''
    (OUT/'review.html').write_text(template.replace('__PACKET__',data),encoding='utf-8')
    verify(report['pins'])
    save(OUT/'page_build.json',dict(status='built_not_browser_accepted',photo_count=30,
        starts_with_all_human_observations_unknown=True,no_AI_to_human_autofill=True,
        human_export_is_self_reported_not_physical_GT=True,requires_no_server=True,
        pins={str(Path(__file__)):digest(Path(__file__)),str(OUT/'review.html'):digest(OUT/'review.html')}))
    print('Offline30-photo review page generated; all human fields unknown, no automatic acceptance.')


if __name__=='__main__':main()
