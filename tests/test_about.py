"""關於對話框：整合版與三個組件的版號、GitHub 連結。"""

from PySide6.QtWidgets import QLabel

from apu_astro import __version__, components


def test_about_lists_component_versions(app, window):
    names = [a.text() for a in window._version_menu().actions()]
    assert any("APU Astro" in n for n in names)
    window.show_about()
    box = window._about
    text = box.text()
    assert __version__ in text and components.REPO_URL in text
    for name, version, url in components.versions():
        assert f"APU {name}" in text and version in text and url in text
    assert any(lbl.openExternalLinks() for lbl in box.findChildren(QLabel))
    box.close()


def test_component_line():
    line = components.line()
    assert line.startswith("Pick ") and "・Photons " in line and "・Processing " in line
