"""Generic distinct-detection evidence identity; same original IoU threshold."""
import copy
from inspection_agent.port_tiling import box_iou
from port_semantic_support_policy import complete


def unique_components(additions,teacher,student):
    if (teacher['source_sha256']!=student['source_sha256'] or
        teacher['predictions']['source_shape']!=student['predictions']['source_shape'] or
        teacher['weight_sha256']==student['weight_sha256']):raise ValueError('source detector identity mismatch')
    shape=teacher['predictions']['source_shape'];nodes=[]
    for model in (teacher,student):
        for index,row in enumerate(model['predictions']['merged_predictions']):
            if row['confidence']>.05 and complete(row,shape):nodes.append((model['weight_sha256'],index,row))
    parent=list(range(len(nodes)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    edges=[]
    for i,(digest,_,row) in enumerate(nodes):
        for j in range(i+1,len(nodes)):
            other_digest,_,other=nodes[j]
            if digest!=other_digest and row['class_id']==other['class_id'] and box_iou(row['box_xyxy'],other['box_xyxy'])>=.5:
                a,b=find(i),find(j);parent[max(a,b)]=min(a,b);edges.append((i,j))
    groups={}
    for i,node in enumerate(nodes):groups.setdefault(find(i),[]).append([node[0],node[1]])
    linked=[]
    for row in additions:
        digest=row['semantic_detector_weight_sha256'];original_score=row['proposal_detector_score']
        matches=[i for i,(owner,_,native) in enumerate(nodes) if owner==digest and native['class_id']==row['class_id']
            and abs(native['confidence']-original_score)<=1e-6 and all(abs(a-b)<=1e-6 for a,b in zip(native['box_xyxy'],row['box_xyxy']))]
        if len(matches)!=1:raise ValueError('unlinked or ambiguous original proposal node')
        i=matches[0];component=find(i)
        if len({n[0] for n in groups[component]})!=2:raise ValueError('component lacks two distinct detector checkpoints')
        linked.append((component,row))
    linked.sort(key=lambda pair:(-pair[1]['confidence'],*pair[1]['box_xyxy'],pair[1]['class_id']))
    used=set();kept=[];audit=[]
    for component,row in linked:
        suppressed=component in used
        audit.append(dict(box_xyxy=row['box_xyxy'],class_id=row['class_id'],component_members=groups[component],
                          suppressed_shared_detection_component=suppressed))
        if suppressed:continue
        used.add(component);kept.append(copy.deepcopy(row))
    return kept,audit
