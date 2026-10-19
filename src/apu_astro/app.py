"""啟動：python -m apu_astro [檔案或資料夾...]，或打包好的 APUAstro.exe。"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # 打包成沒有主控台的 exe 時 stdout / stderr 是 None，有人 print 就會當掉
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("APU Astro")
    app.setOrganizationName("APU")
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass

    from .i18n import LANGUAGES, set_language
    from .settings import load_settings
    from .theme import apply_theme
    from .window import MainWindow

    st = load_settings()
    if st.get("language") in LANGUAGES:
        set_language(st["language"])
    apply_theme(app)
    win = MainWindow([a for a in argv if not a.startswith("-")])
    screen = app.primaryScreen().availableGeometry()
    w, h = st.get("geometry") or (1440, 920)
    win.resize(min(int(w), screen.width() - 40), min(int(h), screen.height() - 60))
    if st.get("maximized"):
        win.showMaximized()
    else:
        win.show()
    return app.exec()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
