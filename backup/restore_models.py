"""Restore the public Release model assets without overwriting different local files."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()

def verified(path, expected):
    if digest(path)!=expected:raise ValueError('Checksum mismatch: '+path.name)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'backup/downloads')
    parser.add_argument('--download',action='store_true',help='Download missing files from the public repository Release')
    args=parser.parse_args();assets=args.assets.resolve();assets.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'backup/model-manifest.json').read_text(encoding='utf-8'))
    archive_spec=manifest['model_archive'];specs=[archive_spec,*manifest['sam3']['parts']]
    for spec in specs:
        path=assets/spec['name']
        if not path.exists():
            if not args.download:raise FileNotFoundError(str(path)+'; download Release files here or use --download')
            url=manifest['repository']+'/releases/download/'+manifest['release_tag']+'/'+spec['name']
            temp=path.with_suffix(path.suffix+'.download')
            if temp.exists():raise FileExistsError('Preserve existing partial download: '+str(temp))
            print('Downloading '+spec['name'],flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'WireMind-model-restore'})
            with urllib.request.urlopen(req,timeout=120) as src,temp.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
            verified(temp,spec['sha256']);temp.rename(path)
        verified(path,spec['sha256'])
    allowed={item['path']:item for item in archive_spec['entries']}
    with zipfile.ZipFile(assets/archive_spec['name']) as archive:
        for item in archive.infolist():
            target=(ROOT/item.filename).resolve()
            if not target.is_relative_to(ROOT):raise ValueError('Unsafe archive path')
            expected=allowed.get(item.filename)
            if not expected and not item.filename.startswith('licenses/'):raise ValueError('Unexpected archive member')
            if target.exists():
                if expected:verified(target,expected['sha256'])
                elif target.read_bytes()!=archive.read(item):raise FileExistsError('Existing license differs')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            fd,temp_name=tempfile.mkstemp(prefix=target.name+'.restore-',dir=target.parent);os.close(fd)
            temp=Path(temp_name)
            try:
                with archive.open(item) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
                if expected:verified(temp,expected['sha256'])
                temp.rename(target)
            finally:
                if temp.exists():temp.unlink()
    for alias in manifest.get('model_aliases',[]):
        source=(ROOT/alias['source']).resolve();target=(ROOT/alias['path']).resolve()
        if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):raise ValueError('Unsafe model alias path')
        expected=allowed.get(alias['source'])
        if not expected or expected['sha256']!=alias['sha256']:raise ValueError('Unverified alias source')
        verified(source,alias['sha256'])
        if target.exists():verified(target,alias['sha256']);continue
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temp_name=tempfile.mkstemp(prefix=target.name+'.restore-',dir=target.parent);os.close(fd);temp=Path(temp_name)
        try:
            shutil.copyfile(source,temp);verified(temp,alias['sha256']);temp.rename(target)
        finally:
            if temp.exists():temp.unlink()
    sam=manifest['sam3'];target=(ROOT/sam['path']).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Unsafe SAM path')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():verified(target,sam['sha256'])
    else:
        fd,temp_name=tempfile.mkstemp(prefix='sam3.restore-',dir=target.parent);os.close(fd);temp=Path(temp_name)
        try:
            with temp.open('wb') as dst:
                for part in sam['parts']:
                    with (assets/part['name']).open('rb') as src:shutil.copyfileobj(src,dst,8*1024*1024)
            if temp.stat().st_size!=sam['bytes']:raise ValueError('SAM size mismatch')
            verified(temp,sam['sha256']);temp.rename(target)
        finally:
            if temp.exists():temp.unlink()
    print('All model files restored and SHA-256 verified; experimental head is NOT enabled.')

if __name__=='__main__':main()
