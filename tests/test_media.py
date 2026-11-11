import json
from types import SimpleNamespace

import pytest

from magpie_wan import media


def test_ffprobe_measures_actual_video_specifications(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(media.shutil, "which", lambda name: name)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "format": {"duration": "5.0625"}, "streams": [{"codec_type": "video", "width": 1280,
            "height": 736, "avg_frame_rate": "16/1", "nb_frames": "81"}]}))
    monkeypatch.setattr(media.subprocess, "run", run)
    measured = media.validate_video_file(video)
    assert measured == {"duration": 5.0625, "streams": ["video"], "width": 1280, "height": 736,
                        "fps": 16.0, "frame_count": 81}
    assert "avg_frame_rate" in calls[0][calls[0].index("-show_entries") + 1]


@pytest.mark.parametrize("rate", ["0/0", "N/A", None])
def test_unknown_frame_rate_stays_unknown(tmp_path, monkeypatch, rate):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(media.shutil, "which", lambda name: name)
    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr="",
        stdout=json.dumps({"format": {"duration": "1"}, "streams": [{"codec_type": "video", "avg_frame_rate": rate}]})))
    assert media.validate_video_file(video)["fps"] is None
