"""HTTP storage/security tests use only generated images, not actual annotations."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from core import image_binding, sha256
from serve_human_bundle_recheck import handler
import test_human_bundle_recheck as fixtures


class HumanServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        (self.out/'images').mkdir()
        for p in ['source.jpg','reference.jpg','images/fixture.jpg','images/reference.jpg']:
            Image.new('RGB',(200,100),'white').save(self.out/p)
        scope,c,self.s=fixtures.HumanRecheckTests().fixture()
        binding=image_binding(self.out/'source.jpg')
        scope.update(reference_binding=binding,reference_image_path=str(self.out/'reference.jpg'))
        c.update(image_binding=binding,source_path=str(self.out/'source.jpg'))
        self.s['image_binding']=binding
        pin=self.out/'source_report.json';pin.write_text('{}',encoding='utf-8')
        self.catalog={'scope':scope,'cases':[c],'source_pins':{str(pin):sha256(pin)},'source_report_sha256':sha256(pin)}
        self.server=HTTPServer(('127.0.0.1',0),handler(self.catalog,self.out,'fixture_token',0))
        port=self.server.server_address[1]
        self.server.RequestHandlerClass=handler(self.catalog,self.out,'fixture_token',port)
        self.url='http://127.0.0.1:'+str(port)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def post(self,value,token='fixture_token',origin=None):
        request=Request(self.url+'/recheck',data=json.dumps(value).encode(),headers={
            'Content-Type':'application/json','X-Review-Token':token,'Origin':origin or self.url})
        try:
            with urlopen(request) as r:return r.status,json.load(r)
        except HTTPError as e:return e.code,json.load(e)

    def test_append_only_reports_replay_and_original_unchanged(self):
        original=json.loads(json.dumps(self.catalog))
        paths=[]
        for _ in range(2):
            status,result=self.post(self.s);self.assertEqual(status,200)
            paths.append(result['saved_report']);saved=json.loads(Path(paths[-1]).read_text(encoding='utf-8'))
            self.assertEqual(saved,result['report']);self.assertEqual(saved['evidence_source'],'software_fixture')
            self.assertEqual(saved['observed_visible_attachment'],{'visible_lead_emergence':'A','wire_entry_socket':'B'})
            self.assertIn('软件测试（非真实验收）',Path(paths[-1]).with_suffix('.md').read_text(encoding='utf-8'))
        self.assertNotEqual(*paths);self.assertEqual(self.catalog,original)

    def test_security_and_malformed_input_rejected(self):
        self.assertEqual(self.post(self.s,token='bad')[0],403)
        self.assertEqual(self.post(self.s,origin='http://external.invalid')[0],403)
        for value in [[],None,{'case_id':'other'}]:self.assertEqual(self.post(value)[0],400)
        self.assertEqual(list(self.out.glob('review_*')),[])

    def test_changed_photo_or_report_blocks_save(self):
        Image.new('RGB',(200,100),'black').save(self.out/'images/fixture.jpg')
        self.assertEqual(self.post(self.s)[0],400)
        Image.new('RGB',(200,100),'white').save(self.out/'images/fixture.jpg')
        (self.out/'source_report.json').write_text('{"changed":true}',encoding='utf-8')
        self.assertEqual(self.post(self.s)[0],400)
        self.assertEqual(list(self.out.glob('review_*')),[])


if __name__=='__main__':unittest.main()
