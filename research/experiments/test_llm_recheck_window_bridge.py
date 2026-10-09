"""Offline adapter check using the two actual returned plans; no GUI or API run."""
import json,unittest
from pathlib import Path
import launch_llm_recheck_window_20261008 as adapter

class BridgeTests(unittest.TestCase):
    def test_actual_results_render(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/llm_recheck_planner_20261008'
        for name,count in [('cabinet2',2),('cabinet4_right1',3)]:
            result=json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
            text=adapter.render_plan(result)
            self.assertEqual(text.count('补证问题：'),count)
            self.assertEqual(text.count('建议核查：'),count)
            self.assertIn(adapter.planner.NOTICE,text)
    def test_safe_fallback(self):
        self.assertIn('原本地流程',adapter.render_plan({'status':'fallback_local_review'}))

if __name__=='__main__':unittest.main()
