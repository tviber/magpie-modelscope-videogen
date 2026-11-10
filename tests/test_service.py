import math
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from magpie_wan import provider
from magpie_wan.service import ServiceError, WanService, _infer_dimensions, parse_accounts, parse_media_roots


def make_frame(path, size=(320, 180), color=(24, 80, 160)):
    Image.new("RGB", size, color).save(path)
    return path


def make_service(tmp_path, accounts=None):
    return WanService(
        data_dir=tmp_path / "wan-service",
        accounts=accounts or ["http://127.0.0.1:9223"],
        media_roots=[tmp_path],
    )


def test_submit_persists_job_and_deduplicates_explicit_idempotency_key(tmp_path):
    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    body = {
        "prompt": "A figure takes one step forward.",
        "first_frame": str(frame),
        "idempotency_key": "same-request",
    }

    first, created = service.submit(body)
    repeated, repeated_created = service.submit(body)

    assert created is True
    assert repeated_created is False
    assert repeated["id"] == first["id"]
    assert first["status"] == "queued"
    assert first["model"] == "wan2.1-flf2v-14b"
    assert first["input_fingerprint"].startswith("sha256:")
    assert Image.open(service.store.resolve_data_path(first["first_frame"])).size == (320, 180)
    assert not Path(first["first_frame"]).is_absolute()
    assert first["style"] is None
    assert first["negative_prompt"] is None
    assert math.isclose(first["width"] / first["height"], 16 / 9, rel_tol=0.015)

    with pytest.raises(ServiceError, match="different input"):
        service.submit({**body, "prompt": "A different action."})
    service.stop()


def test_account_count_is_not_limited_to_two_and_urls_are_not_persisted(tmp_path):
    endpoints = [f"http://127.0.0.1:{9223 + index}" for index in range(5)]
    service = make_service(tmp_path, endpoints)
    frame = make_frame(tmp_path / "first.png")
    jobs = [service.submit({"prompt": f"Motion {index}", "first_frame": str(frame)})[0]
            for index in range(10)]

    assert service.accounts == [f"account-{index + 1}" for index in range(5)]
    assert [job["account"] for job in jobs[:5]] == service.accounts
    with sqlite3.connect(service.store.db_path) as db:
        persisted_accounts = [row[0] for row in db.execute("SELECT DISTINCT account FROM jobs")]
    assert sorted(persisted_accounts) == sorted(service.accounts)
    assert all("http://" not in account for account in persisted_accounts)
    service.stop()


def test_account_list_parser_accepts_one_or_many_endpoints():
    assert parse_accounts("http://127.0.0.1:9223") == ["http://127.0.0.1:9223"]
    assert len(parse_accounts(",".join(f"http://127.0.0.1:{port}" for port in range(9223, 9231)))) == 8


def test_media_roots_must_be_explicit_and_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("WAN_MEDIA_ROOTS", raising=False)
    with pytest.raises(ServiceError, match="Set WAN_MEDIA_ROOTS"):
        parse_media_roots()
    with pytest.raises(ServiceError, match="existing directories"):
        parse_media_roots(str(tmp_path / "missing"))
    assert parse_media_roots(str(tmp_path)) == [tmp_path.resolve()]


def test_default_dimensions_keep_common_aspect_ratio(tmp_path):
    frame = make_frame(tmp_path / "wide.png", (320, 180))

    width, height = _infer_dimensions(frame)

    assert width % 16 == height % 16 == 0
    assert math.isclose(width / height, 16 / 9, rel_tol=0.015)


def test_modelscope_request_sends_prompt_verbatim_unless_style_is_explicit():
    client = provider.MS.__new__(provider.MS)
    client.default_size = None
    client.models = {"WAN_VIDEO_2_1_FLF2V_14_B": {"versionId": "version-1"}}

    request = client.request("WAN_VIDEO_2_1_FLF2V_14_B", "first-url", "last-url", "move slowly", 5,
                             width=1280, height=720)

    assert request["promptArgs"] == {"prompt": "move slowly"}
    assert request["basicDiffusionArgs"]["width"] == 1280
    assert request["basicDiffusionArgs"]["height"] == 720
    assert request["extraImageInputFrontArgs"]["image"] == "last-url"


def test_public_job_uses_generic_account_label_and_no_local_path(tmp_path):
    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    job, _ = service.submit({"prompt": "A short movement.", "first_frame": str(frame)})
    output = service.store.jobs_dir / job["id"] / "video.mp4"
    relative_output = service.store.relative_data_path(output)
    service.store.update(job["id"], status="completed", result_path=relative_output)

    public = service.public_job(service.store.get(job["id"]))

    assert public["account"] == "account-1"
    assert "video_path" not in public
    assert public["output_path"].endswith("/video.mp4")
    assert not Path(public["output_path"]).is_absolute()
    assert "first.png" not in str(public)
    assert public["content_url"].endswith(f"/{job['id']}/content")
    service.stop()


def test_removing_an_account_with_active_jobs_fails_clearly(tmp_path):
    endpoints = ["http://127.0.0.1:9223", "http://127.0.0.1:9224"]
    service = make_service(tmp_path, endpoints)
    frame = make_frame(tmp_path / "first.png")
    service.submit({"prompt": "First motion", "first_frame": str(frame)})
    service.submit({"prompt": "Second motion", "first_frame": str(frame)})
    service.stop()

    with pytest.raises(ServiceError, match="account-2"):
        make_service(tmp_path, endpoints[:1])


def test_named_accounts_survive_reorder_and_port_changes(tmp_path):
    service = make_service(tmp_path, {"primary": "http://localhost:9223", "secondary": "http://localhost:9224"})
    service.store.verify_identity("primary", "Id:123")
    service.store.verify_identity("secondary", "Id:456")
    frame = make_frame(tmp_path / "first.png")
    job, _ = service.submit({"prompt": "motion", "first_frame": str(frame)})
    service.stop()
    service = make_service(tmp_path, {"secondary": "http://localhost:9334", "primary": "http://localhost:9333"})
    service.store.verify_identity("primary", "Id:123")
    assert service.store.get(job["id"])["account"] == "primary"
    with pytest.raises(ServiceError, match="identity mismatch"):
        service.store.verify_identity("primary", "Id:456")
    with sqlite3.connect(service.store.db_path) as db:
        rows = str(db.execute("SELECT * FROM accounts").fetchall())
    assert "Id:123" not in rows and "localhost" not in rows
    service.stop()


def test_legacy_jobs_require_explicit_initial_binding(tmp_path):
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / "first.png")
    service.submit({"prompt": "motion", "first_frame": str(frame)})
    with sqlite3.connect(service.store.db_path) as db:
        db.execute("DELETE FROM accounts")  # Simulate the pre-fingerprint database.
    service.stop()
    service = make_service(tmp_path)
    with pytest.raises(ServiceError, match="explicit --bind-accounts"):
        service.store.verify_identity("account-1", "Id:123")
    service.store.verify_identity("account-1", "Id:123", bind_legacy=True)
    with pytest.raises(ServiceError, match="identity mismatch"):
        service.store.verify_identity("account-1", "Id:456", bind_legacy=True)
    service.stop()


def test_reordered_legacy_urls_cannot_silently_change_logins(tmp_path):
    urls = ["http://localhost:9223", "http://localhost:9224"]
    service = make_service(tmp_path, urls)
    service.store.verify_identity("account-1", "Id:123")
    with pytest.raises(ServiceError, match="already bound"):
        service.store.verify_identity("account-2", "Id:123")
    service.stop()
    service = make_service(tmp_path, urls[::-1])
    with pytest.raises(ServiceError, match="identity mismatch"):
        service.store.verify_identity("account-1", "Id:456")
    service.stop()


def test_named_account_environment(monkeypatch):
    monkeypatch.setenv("WAN_MODELSCOPE_ACCOUNTS", '{"main":"http://localhost:9223"}')
    assert parse_accounts() == {"main": "http://localhost:9223"}
    monkeypatch.setenv("WAN_MODELSCOPE_ACCOUNTS", '{"../invalid":"http://localhost:9223"}')
    with pytest.raises(ServiceError):
        parse_accounts()


def test_windows_paths_migrate_and_remain_confined(tmp_path):
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / "first.png")
    job, _ = service.submit({"prompt": "motion", "first_frame": str(frame), "last_frame": str(frame)})
    output = service.store.jobs_dir / job["id"] / "video.mp4"
    output.write_bytes(b"offline video fixture")
    service.store.update(job["id"], status="completed",
                         first_frame=job["first_frame"].replace("/", "\\"),
                         last_frame=job["last_frame"].replace("/", "\\"),
                         result_path=service.store.relative_data_path(output).replace("/", "\\"))
    service.stop()
    service = make_service(tmp_path)
    migrated = service.store.get(job["id"])
    for field in ("first_frame", "last_frame", "result_path"):
        assert "\\" not in migrated[field]
        assert service.store.resolve_data_path(migrated[field]).is_file()
    assert service.store.resolve_data_path(migrated["result_path"].replace("/", "\\")) == output
    for path in ("..\\outside", "C:\\data\\video.mp4", "C:video.mp4", "\\\\host\\share\\video.mp4", "/etc/passwd"):
        with pytest.raises(ServiceError, match="invalid"):
            service.store.resolve_data_path(path)
    service.stop()


@pytest.mark.parametrize("status_code,failures", [(401, 1), (403, 1), (500, 3)])
def test_download_failure_refreshes_original_task_and_unblocks_queue(tmp_path, monkeypatch, status_code, failures):
    from unittest.mock import Mock
    import requests
    from magpie_wan import media

    service = make_service(tmp_path)
    frame = make_frame(tmp_path / "first.png")
    job, _ = service.submit({"prompt": "motion", "first_frame": str(frame)})
    next_job, _ = service.submit({"prompt": "next motion", "first_frame": str(frame)})
    service.store.update(job["id"], status="download_pending", provider_task_id="original-task",
                         provider_url="https://example.org/expired.mp4")
    response = requests.Response()
    response.status_code = status_code
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    for _ in range(failures):
        service._download_job(service.store.get(job["id"]))
    assert service.store.get(job["id"])["provider_url"] is None
    service.stop()
    service = make_service(tmp_path)
    ms = Mock()
    ms.status.return_value = {"status": "SUCCEED", "url": "https://example.org/fresh.mp4"}
    response.status_code = 200
    response._content = b"verified fixture"
    monkeypatch.setattr(media, "save_video_bytes", lambda content, path: path.write_bytes(content))
    service._poll_job(ms, service.store.get(job["id"]))
    ms.status.assert_called_once_with("original-task")
    ms.submit.assert_not_called()
    assert get.call_args.args[0] == "https://example.org/fresh.mp4"
    assert service.store.get(job["id"])["status"] == "completed"
    assert service.store.next_for_account("account-1")["id"] == next_job["id"]
    service.stop()


def test_content_route_requires_bearer_and_serves_migrated_video(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    import requests
    from magpie_wan.service import make_handler

    service = make_service(tmp_path)
    frame = make_frame(tmp_path / "first.png")
    job, _ = service.submit({"prompt": "motion", "first_frame": str(frame)})
    video = service.store.jobs_dir / job["id"] / "video.mp4"
    video.write_bytes(b"local video fixture")
    service.store.update(job["id"], status="completed",
                         result_path=service.store.relative_data_path(video).replace("/", "\\"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/videos/{job['id']}/content"
        assert requests.get(url, timeout=3).status_code == 401
        assert requests.get(url + "?token=test-token", timeout=3).status_code == 401
        response = requests.get(url, headers={"Authorization": "Bearer test-token"}, timeout=3)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "video/mp4"
        assert response.content == video.read_bytes()
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        service.stop()
