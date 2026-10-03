"""Replay existing real-image evidence into an auditable topology demo."""
from pathlib import Path
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.project.resolve(), args.output.resolve()
    sys.path.insert(0, str(root))
    from inspection_agent import ConnectionGraph, connection_graph_from_sam_endpoints
    from inspection_agent.workflow import InspectionTask

    visual_path = root / "output/template_reviews/20260827_160916/report.json"
    cache = visual_path.parent / "sam3/inspection"
    endpoints_path = out / "guarded_endpoints/endpoint_report.json"
    map_path = root / "config/topology_demo/wrong2_visible_anchor_map.example.json"
    expected_path = root / "config/topology_demo/wrong2_visible_anchor_expected.example.json"
    visual, endpoints, terminal_map = read(visual_path), read(endpoints_path), read(map_path)
    observed, adapter = connection_graph_from_sam_endpoints(endpoints, terminal_map)
    expected = ConnectionGraph.from_dict(read(expected_path))
    task = InspectionTask("cabinet-cache-real-image-probe-20260930", observed.scene_type,
                          visual["reference"], visual["inspection"])
    task.record_visual_analysis(visual, tool_name="cached_cabinet_dino_sam3_review",
                                source_report=visual_path)
    comparison = task.assess_topology(expected, observed, minimum_confidence=0.75)
    report = task.to_report()
    report["execution"] = {"mode": "cached_real_image_replay", "fresh_sam_inference": False,
                           "historical_source_and_checkpoint_identity_verified": False,
                           "declared_nodes": "visible anchors, not verified electrical terminals",
                           "expected_edges": "pre-existing visible-anchor contract, not wiring ground truth"}
    report["sam_adapter"] = adapter
    report["observed_graph"] = observed.to_dict()
    paths = [visual_path, cache / "input.jpg", cache / "report.json", endpoints_path,
             map_path, expected_path, Path(__file__).resolve(),
             root / "inspection_agent/sam_topology_adapter.py",
             root / "inspection_agent/topology.py",
             root / "manual_review/extract_sam3_visible_segment_endpoints.py"]
    paths.extend(sorted(cache.glob("mask_[0-9]*.png")))
    report["current_artifact_fingerprints"] = [fingerprint(p) for p in paths]
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_task.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    states = Counter(r["state"] for r in adapter["evidence_records"])
    images = []
    for title, path in [("原视觉流程：2 个差异复核候选", visual_path.parent / "sam3_fusion_boxes.jpg"),
                        ("真实图的 SAM 端点：红蓝点表示可见线段两端", out / "guarded_endpoints/endpoints_overlay.jpg")]:
        uri = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{uri}"></figure>')
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(states.items()))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>机柜实图拓扑尝试</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.65 system-ui;background:#f5f6f8;color:#202933}h1{font-size:26px}figure{margin:24px 0;padding:16px;background:white;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse}td{border:1px solid #ccc;padding:8px 16px}.status{padding:16px;background:#fff0cf;border-radius:8px}</style>
<h1>机柜实图：视觉证据 → 候选关系 → 人工复核</h1>
<p class="status">本次结果：证据不足，任务停留在待人工复核。</p>
<p>使用现有真实机柜照片及旧 SAM 缓存，重新提取带分支检查的端点。没有重新执行大模型推理。88 个线段分量中 77 个符合两端且无分支的条件，11 个弃权。</p>
<p>沿用此前人工声明的两个可见锚点，得到 1 条候选关系：m004_c01，分数 0.61328125。固定门槛 0.75，未通过。其余 87 个分量没有形成候选关系。</p>
<p>这两个锚点没有经电气端子身份确认；期望关系是原演示约定。分数来自 SAM 掩膜，不代表校准过的接线正确概率。本次可展示证据流转和弃权，仍需补齐端子身份与预期接线表，才能评价真实接线拓扑。</p>
''' + "".join(images) + '<h2>适配器记录</h2><table>' + rows + '''</table>
<p>原视觉复核的 2 个差异候选与拓扑候选分别保存。任务未写入人工确认、维修指令或复检完成记录。</p>
<p>下一步视觉模块仍负责定位差异和端口；拓扑模块使用可确认的端子区域及独立可见线段建立候选关系。端口裁块模型先通过可选单图入口接入，沿用已冻结规则与回退条件。</p></html>'''
    (out / "demo.html").write_text(page, encoding="utf-8")
    assert comparison["decision"] == "insufficient_evidence"
    assert adapter["summary"]["emitted_connection_count"] == 1
    assert adapter["summary"]["legacy_geometry_unverified_record_count"] == 0
    assert report["state"] == "awaiting_human_review"
    assert not report["human_conclusions"] and not report["repair_guidance"]
    saved = read(out / "agent_task.json")
    assert all(fingerprint(Path(p["path"])) == p for p in saved["current_artifact_fingerprints"])
    print(json.dumps({"decision": comparison["decision"], "state": report["state"],
                      "visual_candidate_count": report["machine_evidence"][0]["candidate_count"],
                      "adapter": adapter["summary"], "states": states,
                      "fingerprints_verified": len(paths), "demo": str(out / "demo.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
