"""Fixed launcher for assembly_anchor_review_app anchor-click callbacks."""

from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import assembly_anchor_review_app as base


class AnchorCanvas(base.AnchorCanvas):
    """Routes click completion to the top-level AnchorPicker, not scroll viewport."""
    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and self.add_mode and len(self.points) < 4:
            point = event.pos()
            if self.rect().contains(point):
                self.points.append(point)
                self.add_mode = False
                self.render()
                picker = self.window()
                if hasattr(picker, "anchor_added"):
                    picker.anchor_added()


def main() -> None:
    base.AnchorCanvas = AnchorCanvas
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = base.AnchorReviewWindow()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
