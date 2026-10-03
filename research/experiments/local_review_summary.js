// Read-only draft summary: no form mutation and no inferred connection edges.
const maps=JSON.parse(document.getElementById('maps').textContent);
const labels={conflicting_support:'支持存在冲突',no_visible_candidate:'没有可见候选（不判缺线）',pending_candidates:'候选仍待处理',uncertain_evidence:'有无法确认项',all_candidates_rejected_not_missing_wire:'候选全部排除（不判缺线）',local_support_not_connection:'局部支持，不是连接',port_identity_unconfirmed:'端子身份未确认',no_verified_complete_cable_observations:'缺完整线缆两端及身份观察',test_record_not_operator_evidence:'测试记录，不是人工确认',expected_scope_unconfirmed:'预期范围未确认',expected_connections_unknown:'预期连接表未知',expected_table_unconfirmed:'预期表及依据未确认',conflicting_local_support:'局部支持冲突'};
const panel=node('section');panel.id='review-summary';panel.className='card';cards.before(panel);
function draftSummary(){
 const rows=bundle.items.map(item=>{const c=controls.get(item.item_id);return {item_id:item.item_id,title:item.title,case:item.case,candidate_reviews:[...c.pairs].map(([rid,e])=>({record_id:rid,state:e.select.value,evidence_note:e.note.value.trim()})),entry_review:{state:c.entry.select.value,evidence_note:c.entry.note.value.trim()},reasons:[]}});
 const usage=new Map();for(const row of rows)for(const c of row.candidate_reviews.filter(c=>c.state==='supported')){const key=JSON.stringify([row.case,c.record_id]);usage.set(key,(usage.get(key)||0)+1)}
 for(const row of rows){const counts=Object.fromEntries(Object.keys(states).map(s=>[s,row.candidate_reviews.filter(c=>c.state===s).length]));row.candidate_counts=counts;
 if(counts.supported>1||row.candidate_reviews.some(c=>c.state==='supported'&&usage.get(JSON.stringify([row.case,c.record_id]))>1))row.reasons.push('conflicting_support');
 if(!row.candidate_reviews.length)row.reasons.push('no_visible_candidate');if(counts.pending)row.reasons.push('pending_candidates');if(counts.uncertain||row.entry_review.state==='uncertain')row.reasons.push('uncertain_evidence');
 if(row.candidate_reviews.length&&counts.rejected===row.candidate_reviews.length)row.reasons.push('all_candidates_rejected_not_missing_wire');if(counts.supported)row.reasons.push('local_support_not_connection');
 const item=bundle.items.find(i=>i.item_id===row.item_id),ports=maps[row.case].ports.filter(p=>p.id===item.port_id);if(ports.length!==1||ports[0].confirmed!==true)row.reasons.push('port_identity_unconfirmed');
 }
 const topology_readiness=Object.keys(maps).sort().map(caseId=>{const m=maps[caseId],rs=rows.filter(r=>r.case===caseId),blockers=['no_verified_complete_cable_observations'];if(document.getElementById('mode').value==='automation_fixture')blockers.push('test_record_not_operator_evidence');if(rs.some(r=>r.reasons.includes('port_identity_unconfirmed')))blockers.push('port_identity_unconfirmed');if(m.scope.expected_complete!==true)blockers.push('expected_scope_unconfirmed');if(!Array.isArray(m.expected_connections))blockers.push('expected_connections_unknown');if(m.expected_review.confirmed!==true||!['reviewer','evidence_note'].every(k=>typeof m.expected_review[k]==='string'&&m.expected_review[k].trim()))blockers.push('expected_table_unconfirmed');if(rs.some(r=>r.reasons.includes('conflicting_support')))blockers.push('conflicting_local_support');return {case:caseId,decision:'insufficient_evidence',blockers,comparison_performed:false,connection_edges:[]}});
 return {entries:rows,topology_readiness,connection_edges:[],automatic_fault_verdict:false};
}
function jump(itemId,rid){const c=controls.get(itemId),e=rid?c.pairs.get(rid):c.entry;if(rid)e.wrapper.closest('details').open=true;e.wrapper.scrollIntoView({block:'center'});e.select.focus({preventScroll:true})}
function renderSummary(){const report=draftSummary();panel.replaceChildren(node('h2','实时复核汇总与待办'));
 let valid=true;try{build()}catch(e){valid=false;panel.append(node('p','当前填写尚未通过保存校验：'+e.message))}
 panel.dataset.valid=String(valid);panel.append(node('p','此汇总读取当前填写，不代表已保存或来源实时核验。连接边始终为0。'));
 const counts=Object.fromEntries(Object.keys(states).map(s=>[s,report.entries.reduce((n,r)=>n+r.candidate_counts[s],0)]));panel.append(node('p','支持 '+counts.supported+' / 排除 '+counts.rejected+' / 无法确认 '+counts.uncertain+' / 未处理 '+counts.pending));
 for(const gate of report.topology_readiness){panel.append(node('h3',gate.case+'：证据不足，未比较拓扑'),node('p',gate.blockers.map(b=>labels[b]).join('；')))}
 for(const row of report.entries){const d=node('div');d.className='queue-row';d.append(node('h3',row.title),node('p',row.reasons.map(r=>labels[r]).join('；')));
 const add=(rid,text)=>{const b=node('button',text);b.type='button';b.dataset.itemId=row.item_id;if(rid)b.dataset.recordId=rid;b.onclick=()=>jump(row.item_id,rid);d.append(b)};
 add(null,'定位入口备注');for(const c of row.candidate_reviews){const bad=(c.state==='pending')!==(c.evidence_note==='');if(c.state==='pending'||c.state==='uncertain'||bad||row.reasons.includes('conflicting_support'))add(c.record_id,'定位 '+c.record_id+' · '+states[c.state]+(bad?'（依据未通过校验）':''))}panel.append(d)}
 panel.dataset.report=JSON.stringify(report);
}
document.addEventListener('input',renderSummary);document.addEventListener('change',renderSummary);
document.getElementById('apply').addEventListener('click',renderSummary);renderSummary();
