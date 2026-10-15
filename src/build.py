"""Baut airbox_stroemung.html aus ui.html und engine.js (Rechenkern wird als Text eingebettet)."""
import pathlib
here = pathlib.Path(__file__).parent
ui = (here / "ui.html").read_text(encoding="utf-8")
engine = (here / "engine.js").read_text(encoding="utf-8")
assert "</script" not in engine, "engine.js darf kein </script enthalten"
(here.parent / "airbox_stroemung.html").write_text(ui.replace("__ENGINE__", engine), encoding="utf-8")
print("airbox_stroemung.html gebaut")
