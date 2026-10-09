"""Revise only the overview, preserving all other submitted material content."""
from pathlib import Path
import ast, importlib.util, json, hashlib, zipfile, shutil

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'experiments/build_competition_materials_20261008.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
spec=importlib.util.spec_from_file_location('materials_builder',source)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
env={'TITLE':builder.TITLE}
for n in main.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'summary','innovation'} for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(source),'exec'),env)
assert len(env['innovation'])<=300
call=next(n.value for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='pdf' and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='04_作品信息/作品信息概要表.pdf')
target=builder.OUT/'04_作品信息/作品信息概要表.pdf'
archive=builder.BUILD/'overview_before_innovation_revision.pdf'
if not archive.exists(): shutil.copyfile(target,archive)
args=[eval(compile(ast.Expression(a),str(source),'eval'),env) for a in call.args]
builder.pdf(*args)
assert len(builder.PdfReader(target).pages)==1
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(dest,folder):
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(folder).as_posix())
    with zipfile.ZipFile(dest) as z: assert z.testzip() is None
pack(ROOT/'delivery/智接_WireMind_材料可编辑文字稿_20261008.zip',builder.BUILD/'editable')
check=builder.OUT/'文件校验清单.json'
data=json.loads(check.read_text(encoding='utf-8'))
for item in data['files']:
    if item['path']=='04_作品信息/作品信息概要表.pdf': item.update(bytes=target.stat().st_size,sha256=sha(target))
builder.save(check,data)
bundle=ROOT/'delivery/智接_WireMind_参赛材料_待填写签字_20261008.zip'
pack(bundle,builder.OUT)
receipt=builder.BUILD/'final_delivery_verification.json'
data=json.loads(receipt.read_text(encoding='utf-8'))
data.update(bytes=bundle.stat().st_size,sha256=sha(bundle),overview_innovation_revised=True)
builder.save(receipt,data)
builder.save(builder.BUILD/'overview_innovation_revision.json',{'innovation_chars':len(env['innovation']),'pages':1,'personal_fields_blank':True,'other_materials_rewritten':False,'zip_crc':'pass'})
print(json.dumps({'innovation_chars':len(env['innovation']),'pages':1,'archives_updated':True}))
