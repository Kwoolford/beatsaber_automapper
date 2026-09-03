"""P4b (2026-09-02): the compete test — a blind pair carries no tell, the key is the only
reader of the roles, a loss with a reason becomes a ledger entry AND a bench row, and the
table prints a win rate."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import types
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("compete", REPO / "scripts" / "compete.py")


def _zip(path: pathlib.Path, song: str, author: str, diffs: list[str], notes: int,
         audio_name: str = "song.egg", version="2.0.0"):
    info = {"_version": version, "_songName": song, "_songSubName": "feat. X",
            "_songAuthorName": "band", "_levelAuthorName": author, "_beatsPerMinute": 160,
            "_previewStartTime": 61, "_previewDuration": 45, "_songFilename": audio_name,
            "_coverImageFilename": "cover.jpg", "_environmentName": "NiceEnvironment",
            "_songTimeOffset": 0, "_customData": {"_editors": {"ChroMapper": {}}},
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                                        "_difficultyBeatmaps": [
                                            {"_difficulty": d, "_difficultyRank": 7,
                                             "_beatmapFilename": f"{d}Standard.dat",
                                             "_noteJumpMovementSpeed": 18,
                                             "_noteJumpStartBeatOffset": 0,
                                             "_customData": {"_difficultyLabel": author,
                                                             "_requirements": []}}
                                            for d in diffs]}]}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Info.dat", json.dumps(info))
        zf.writestr("cover.jpg", b"jpg")
        zf.writestr(audio_name, b"OggS-audio")
        for d in diffs:
            bm = {"_version": "2.2.0",
                  "_notes": [{"_time": i, "_lineIndex": 0, "_lineLayer": 0, "_type": i % 2,
                              "_cutDirection": 8} for i in range(notes)],
                  "_obstacles": [], "_events": [],
                  "_customData": {"_bookmarks": [{"_name": author}], "_time": 999}}
            zf.writestr(f"{d}Standard.dat", json.dumps(bm))


def _redirect(tmp: pathlib.Path):
    C.STAGE = tmp / "for_review" / "compete"
    C.KEY = C.STAGE / ".key.json"
    C.CATALOG = tmp / "reviewed"
    C.LEDGER = tmp / "ledger.json"
    C.BENCH = tmp / "bench.json"
    C.HUMAN = tmp / "raw"
    C.HUMAN.mkdir(parents=True)


def test_blind_pair_has_no_tell_and_verdict_loss_becomes_bench_row(tmp_path):
    _redirect(tmp_path)
    _zip(C.HUMAN / "ab123.zip", "Real Title", "famousmapper", ["Expert", "ExpertPlus"], 40)
    ours = tmp_path / "LOOP__ab123.zip"
    _zip(ours, "AGENT ab123", "beatsaber_automapper", ["Expert"], 30, audio_name="song.ogg",
         version="2.1.0")

    assert C.stage_one("ab123", ours, force=False, seed=1, with_page=False) == 0
    key = json.loads(C.KEY.read_text())["ab123"]
    roles = {key["blind"][L]["role"] for L in "XY"}
    assert roles == {"OURS", "HUMAN"}
    infos = {}
    for L in "XY":
        with zipfile.ZipFile(C.STAGE / f"{L}__ab123.zip") as zf:
            assert sorted(zf.namelist()) == ["ExpertStandard.dat", "Info.dat", "song.ogg"]
            infos[L] = json.loads(zf.read("Info.dat"))
            bm = json.loads(zf.read("ExpertStandard.dat"))
            assert "_customData" not in bm                       # bookmarks named the mapper
            blob = zf.read("Info.dat") + zf.read("ExpertStandard.dat")
            assert b"famousmapper" not in blob and b"automapper" not in blob
            assert b"Real Title" not in blob and b"ChroMapper" not in blob
    ix, iy = infos["X"], infos["Y"]
    assert ix["_songName"] == "X ab123" and iy["_songName"] == "Y ab123"
    assert {k: v for k, v in ix.items() if k != "_songName"} == \
           {k: v for k, v in iy.items() if k != "_songName"}      # identical skeleton
    # Expert is chosen over ExpertPlus, like score.load_map / the tutor
    assert key["blind"]["X"]["difficulty"] == "Expert" == key["blind"]["Y"]["difficulty"]
    human_letter = next(L for L in "XY" if key["blind"][L]["role"] == "HUMAN")
    assert key["blind"][human_letter]["notes"] == 40

    a = types.SimpleNamespace(song="ab123", pick=human_letter, because="the drop is late",
                              code="d3", bars="33-34", note="")
    assert C.cmd_verdict(a) == 0
    led = json.loads(C.LEDGER.read_text())["verdicts"]
    assert len(led) == 1 and led[0]["kind"] == "compete" and led[0]["result"] == "loss"
    assert led[0]["better"] == "HUMAN" and led[0]["worse"] == "OURS"
    assert led[0]["maps"]["OURS"].endswith("LOOP__ab123.zip")
    rows = json.loads(C.BENCH.read_text())["rows"]
    assert rows[0]["label"] == "DEFECT" and rows[0]["codes"] == ["D3"]
    assert rows[0]["bars"] == "33-34" and rows[0]["bars_from"] == "kyle"
    assert rows[0]["strength"] == "strong" and rows[0]["quote"] == "the drop is late"
    # unblinded on disk, staging empty
    assert not (C.STAGE / f"X__ab123.zip").exists()
    filed = sorted(p.name for p in C.CATALOG.glob("ab123_*/*.zip"))
    assert any("HUMAN__ab123" in n for n in filed) and any("OURS__ab123" in n for n in filed)
    assert json.loads(C.KEY.read_text())["ab123"]["status"] == "loss"

    out = C.table(C.compete_verdicts(json.loads(C.LEDGER.read_text())))
    assert "WIN RATE 0/1" in out and "[D3] bars 33-34 the drop is late" in out


def test_second_stage_is_refused_and_win_counts(tmp_path):
    _redirect(tmp_path)
    _zip(C.HUMAN / "cd456.zip", "T", "h", ["ExpertPlus"], 20)      # no Expert -> ExpertPlus
    ours = tmp_path / "NOPULSE__cd456.zip"
    _zip(ours, "AGENT", "agent", ["Expert"], 25, audio_name="song.ogg")
    assert C.stage_one("cd456", ours, False, 3, with_page=False) == 0
    assert C.stage_one("cd456", ours, False, 3, with_page=False) == 0    # already staged: no-op
    key = json.loads(C.KEY.read_text())["cd456"]
    hl = next(L for L in "XY" if key["blind"][L]["role"] == "HUMAN")
    assert key["blind"][hl]["difficulty"] == "ExpertPlus"
    ol = "Y" if hl == "X" else "X"
    C.cmd_verdict(types.SimpleNamespace(song="cd456", pick=ol, because="", code=None,
                                        bars=None, note=""))
    out = C.table(C.compete_verdicts(json.loads(C.LEDGER.read_text())))
    assert "WIN RATE 1/1" in out
    assert not C.BENCH.exists()                                   # a win is not a bench row
