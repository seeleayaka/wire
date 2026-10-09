"""Small isolated screenshot diagnostic; no project model or source changes."""
import json
import os
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QLabel

OUT = Path(__file__).resolve().parents[1] / 'artifacts/fresh_four_examples_20261004/font_diagnostic'


def main():
    OUT.mkdir(exist_ok=False)
    app = QApplication([])
    app.setStyle('Fusion')
    initial = QFontDatabase().families()
    rows = []
    for name in ('hidden_default', 'shown_default', 'shown_explicit_font'):
        label = QLabel('Test 中文复核')
        label.resize(280, 65)
        label.setStyleSheet('QLabel {background:white;color:black;}')
        label.setFont(QFont('Microsoft YaHei', 18))
        if name == 'shown_explicit_font':
            font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
            families = QFontDatabase.applicationFontFamilies(font_id)
            assert families, 'explicit local font not loaded'
            label.setFont(QFont(families[0], 18))
        if name != 'hidden_default':
            label.show()
        app.processEvents()
        shot = label.grab()
        shot.save(str(OUT / (name + '.png')))
        pixels = shot.toImage()
        dark = sum(pixels.pixelColor(x, y).lightness() < 100
                   for x in range(pixels.width()) for y in range(pixels.height()))
        rows.append(dict(mode=name,dark_pixels=dark,font_family=label.font().family()))
        label.close()
    report = dict(initial_font_family_count=len(initial),initial_font_families=initial[:8],
                  results=rows,model_inference=False,production_changed=False)
    (OUT / 'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
