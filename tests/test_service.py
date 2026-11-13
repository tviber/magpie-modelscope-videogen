import math
import os
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


@pytest.mark.parametrize("model", ["minimax-h3", "MINIMAX_H3"])
@pytest.mark.parametrize("use_last", [False, True])
def test_minimax_queues_uploads_and_submits_through_existing_worker(tmp_path, model, use_last):
    from unittest.mock import Mock

    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    body = {"model": model, "prompt": "roll slowly", "first_frame": str(frame), "seconds": 3}
    if use_last:
        body["last_frame"] = str(frame)
    job, _ = service.submit(body)
    assert job["model"] == "minimax-h3"
    client = Mock()
    client.upload.side_effect = ["first-url", "last-url"] if use_last else ["first-url"]
    client.request.return_value = {"model": "H3-request"}
    client.estimate.return_value = 68
    client.submit.return_value = "h3-task"
    service._submit_job(client, job)
    saved = service.store.get(job["id"])
    assert saved["status"] == "in_progress"
    assert saved["provider_task_id"] == "h3-task"
    assert service.public_job(saved)["estimate"] == 68
    assert client.request.call_args.args[:3] == ("MINIMAX_H3", "first-url", "last-url" if use_last else None)
    client.submit.assert_called_once_with({"model": "H3-request"})
    service.stop()


def test_wan_i2v_still_rejects_last_frame(tmp_path):
    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    with pytest.raises(ServiceError, match="last_frame"):
        service.submit({"model": "wan2.2-i2v-a14b", "prompt": "move", "first_frame": str(frame),
                        "last_frame": str(frame)})
    service.stop()


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
    assert str(tmp_path) not in str(public)
    assert public["input_frames"][0]["name"] == "first.png"
    assert public["content_url"].endswith(f"/{job['id']}/content")
    service.stop()


def test_public_job_reports_project_request_actual_and_phases(tmp_path):
    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    job, _ = service.submit({"prompt": "Camera move", "first_frame": str(frame), "seconds": 8,
                             "size": "1280x720", "project": "Demo Project One"})
    queued = service.public_job(job)
    assert queued["execution_phase"] == "Waiting for an account worker"
    assert queued["requested"] == {"seconds": 8, "size": "1280x720", "width": 1280, "height": 720, "fps": None}
    assert queued["project"] == "Demo Project One"
    assert queued["prompt"] == "Camera move"
    service.store.update(job["id"], status="completed", actual_metadata='{"duration": 5.06, "width": 1280, "height": 736, "fps": 16.0}')
    actual = service.public_job(service.store.get(job["id"]))["actual"]
    assert actual == {"duration": 5.06, "width": 1280, "height": 736, "fps": 16.0, "frame_count": None}
    assert service.store.list_jobs(project="Demo Project One")[0]["id"] == job["id"]
    assert service.store.list_jobs(project="another") == []
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


def test_launch_browser_account_persists_new_cdp_profile_and_starts_worker(tmp_path, monkeypatch):
    import magpie_wan.service as wan_service

    browser = tmp_path / "chrome"
    browser.write_bytes(b"fixture")
    launches = []
    monkeypatch.setattr(wan_service, "_browser_executable", lambda: browser)
    monkeypatch.setattr(wan_service, "_free_cdp_port", lambda **kwargs: 19456)
    monkeypatch.setattr(wan_service.subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs)))
    service = make_service(tmp_path / "first", {"account-1": "http://127.0.0.1:9223"})
    workers = []
    monkeypatch.setattr(service, "_start_account_worker", workers.append)

    profile = service.launch_browser_account()
    assert profile["id"] == "account-2"
    assert profile["status"] == "connecting"
    assert profile["stale"]
    assert service.account_urls["account-2"] == "http://127.0.0.1:19456"
    assert workers == ["account-2"]
    assert launches and "--remote-debugging-port=19456" in launches[0][0]
    browser_profile = service.store.data_dir / "browser-profiles" / "account-2"
    assert browser_profile.is_dir()
    # Windows chmod/stat do not implement POSIX owner/group permission bits.
    # Keep the browser/profile/registry assertions above and below on every OS.
    if os.name != "nt":
        assert browser_profile.stat().st_mode & 0o777 == 0o700
    service.stop()

    restored = make_service(tmp_path / "first", {"account-1": "http://127.0.0.1:9223"})
    assert restored.account_urls["account-2"] == "http://127.0.0.1:19456"
    assert restored.account_snapshot("account-2")["status"] == "connecting"
    restored.stop()


def test_launch_browser_account_requires_an_installed_browser(tmp_path, monkeypatch):
    import magpie_wan.service as wan_service

    service = make_service(tmp_path)
    monkeypatch.setattr(wan_service, "_browser_executable", lambda: None)
    with pytest.raises(ServiceError, match="Chrome 或 Chromium"):
        service.launch_browser_account()
    service.stop()


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
    monkeypatch.setattr(media, "save_video_bytes", lambda content, path, **_kwargs: path.write_bytes(content))
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


def test_accounts_publish_verified_labels_and_available_magicubes(tmp_path):
    from unittest.mock import Mock
    import time

    service = make_service(tmp_path)
    ms = Mock()
    ms.identity.return_value = "Id:123"
    ms.profile.return_value = "Test user"
    ms.balance.return_value = {"total_balance": 200, "available_balance": 45, "frozen_amount": 155}
    service._refresh_account(ms, "account-1")
    snapshot = service.account_snapshot("account-1")
    assert snapshot["balance"] == 45
    assert snapshot["unit"] == "魔粒"
    assert snapshot["label"] == "Test user"
    assert snapshot["status"] == "connected"
    assert not snapshot["stale"]
    ms.balance.return_value = {"available_balance": 0}
    service._refresh_account(ms, "account-1")
    assert service.account_snapshot("account-1")["balance"] == 0
    ms.balance.return_value = {"unrecognized": 20}
    service._refresh_account(ms, "account-1")
    assert service.account_snapshot("account-1")["balance"] is None
    assert service.account_snapshot("account-1")["balance_error"]
    ms.balance.side_effect = provider.ModelScopeAPIError("/openapi/v1/magicubes/balance", 401)
    service._refresh_account(ms, "account-1")
    snapshot = service.account_snapshot("account-1")
    assert snapshot["balance"] is None
    assert "HTTP 401" in snapshot["balance_error"]
    ms.balance.side_effect = None
    service.account_profiles["account-1"]["checked_at"] = time.time() - 121
    assert service.account_snapshot("account-1")["stale"]
    ms.identity.return_value = "Id:456"
    ms.profile.reset_mock()
    with pytest.raises(ServiceError, match="identity mismatch"):
        service._refresh_account(ms, "account-1")
    ms.profile.assert_not_called()
    service.stop()


def test_project_idempotency_account_selection_and_honest_progress(tmp_path):
    import time
    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path, {"a": "http://localhost:9223", "b": "http://localhost:9224"})
    body = {"prompt": "Camera move", "first_frame": str(frame), "project": "Demo Project", "shot": "Opening",
            "idempotency_key": "opening", "account": "b"}
    job, _ = service.submit(body)
    assert job["account"] == "b"
    assert service.submit(body)[1] is False
    for changes in ({"project": "Other"}, {"shot": "Ending"}, {"account": "a"}):
        with pytest.raises(ServiceError, match="different input"):
            service.submit({**body, **changes})
    public = service.public_job(job)
    assert public["queue_position"] == 1
    assert public["progress"] is None
    assert public["estimated_remaining_seconds"] is None
    service.store.update(job["id"], status="in_progress", last_polled_at=time.time() - 120)
    public = service.public_job(service.store.get(job["id"]))
    assert public["status_stale"]
    assert public["poll_age_seconds"] >= 120
    assert public["progress_kind"] == "indeterminate"
    service.stop()


def test_authenticated_config_preflight_pagination_and_frame_preview(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    import requests
    from magpie_wan.service import make_handler

    frame = make_frame(tmp_path / "first.png")
    service = make_service(tmp_path)
    jobs = [service.submit({"prompt": str(i), "first_frame": str(frame), "project": "Demo Project"})[0]
            for i in range(2)]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        for path in ("/v1/accounts", "/v1/media/roots", "/v1/videos", f"/v1/videos/{jobs[0]['id']}/frames/first_frame"):
            assert requests.get(base + path, timeout=3).status_code == 401
        assert requests.get(base + "/v1/media/roots", headers=headers, timeout=3).json()["roots"] == [str(tmp_path)]
        assert requests.get(base + "/v1/accounts/missing", headers=headers, timeout=3).status_code == 404
        valid = requests.post(base + "/v1/frames/validate", headers=headers, json={"first_frame": str(frame)}, timeout=3)
        assert valid.json()["valid"] is True
        assert len(service.store.list_jobs()) == 2
        first = requests.get(base + "/v1/videos?project=Demo%20Project&limit=1", headers=headers, timeout=3).json()
        second = requests.get(base + f"/v1/videos?project=Demo%20Project&limit=1&offset={first['next_offset']}", headers=headers, timeout=3).json()
        assert first["data"][0]["id"] != second["data"][0]["id"]
        assert second["next_offset"] is None
        preview = requests.get(base + f"/v1/videos/{jobs[0]['id']}/frames/first_frame", headers=headers, timeout=3)
        assert preview.headers["Content-Type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG")
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        service.stop()


def test_old_completed_job_specs_are_backfilled_without_changing_completion_time(tmp_path, monkeypatch):
    from magpie_wan import media
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / "first.png")
    job, _ = service.submit({"prompt": "Motion", "first_frame": str(frame)})
    output = service.store.jobs_dir / job["id"] / "video.mp4"
    output.write_bytes(b"fixture")
    service.store.update(job["id"], status="completed", result_path=service.store.relative_data_path(output))
    old = service.store.get(job["id"])
    monkeypatch.setattr(media, "validate_video_file", lambda _: {"duration": 5.0625, "width": 1280, "height": 736, "fps": 16})
    assert service.job_details(old)["actual"]["fps"] == 16
    assert service.store.get(job["id"])["updated_at"] == old["updated_at"]
    service.stop()


def test_pending_login_is_not_scheduled_or_persisted_and_cancel_closes_browser(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    import magpie_wan.service as module
    service = make_service(tmp_path)
    browser = tmp_path / 'chrome'
    browser.touch()
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(module, '_browser_executable', lambda: browser)
    monkeypatch.setattr(module, '_free_cdp_port', lambda **kwargs: 19457)
    launch = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    account = service.launch_browser_account(pending=True)['id']
    assert '--headless=new' in launch.call_args.args[0]
    assert json.loads((service.store.data_dir / 'browser-accounts.json').read_text()) == {}
    frame = make_frame(tmp_path / 'frame.png')
    with pytest.raises(ServiceError, match='正在登录'):
        service.submit({'prompt': 'motion', 'first_frame': str(frame), 'account': account})
    service.discard_pending_account(account)
    process.terminate.assert_called_once()
    assert account not in service.accounts
    assert not (service.store.data_dir / 'browser-profiles' / account).exists()
    service.stop()
    restored = make_service(tmp_path)
    assert account not in restored.accounts
    restored.stop()


def test_managed_account_avoids_configured_and_existing_browser_ports(tmp_path, monkeypatch):
    import magpie_wan.service as module
    browser = tmp_path / 'chrome'
    browser.touch()
    monkeypatch.setattr(module, '_browser_executable', lambda: browser)
    allocated = []

    def allocate_port(*, excluded=()):
        excluded = set(excluded)
        allocated.append(excluded)
        return next(port for port in (19457, 19458, 19459) if port not in excluded)

    monkeypatch.setattr(module, '_free_cdp_port', allocate_port)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *_args, **_kwargs: None)
    service = make_service(tmp_path, {
        'account-1': 'http://127.0.0.1:19457',
        'account-2': 'http://127.0.0.1:19458',
    })
    service.store.data_dir.joinpath('browser-profiles', 'account-3').mkdir(parents=True)
    service._ensure_managed_browser('account-3')
    assert allocated == [{19457, 19458}]
    assert service.account_urls['account-3'] == 'http://127.0.0.1:19459'
    service.stop()


def test_saved_managed_account_is_reassigned_when_port_became_configured(tmp_path, monkeypatch):
    import json
    import magpie_wan.service as module
    service = make_service(tmp_path, {'account-1': 'http://127.0.0.1:9223'})
    service.store.data_dir.joinpath('browser-profiles', 'account-3').mkdir(parents=True)
    service.store.data_dir.joinpath('browser-accounts.json').write_text(
        json.dumps({'account-3': 'http://127.0.0.1:9223'}), encoding='utf-8')
    service.stop()

    monkeypatch.setattr(module, '_free_cdp_port', lambda **kwargs: 19459)
    restored = make_service(tmp_path, {'account-1': 'http://127.0.0.1:9223'})
    assert restored.account_urls['account-1'] == 'http://127.0.0.1:9223'
    assert restored.account_urls['account-3'] == 'http://127.0.0.1:19459'
    restored.stop()


def test_configured_account_with_private_profile_is_restarted_as_managed(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import magpie_wan.service as module
    browser = tmp_path / 'chrome'
    browser.touch()
    profile = tmp_path / 'wan-service' / 'browser-profiles' / 'account-1'
    profile.mkdir(parents=True)
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(module, '_browser_executable', lambda: browser)
    monkeypatch.setattr(module, '_free_cdp_port', lambda **_kwargs: 19460)
    launch = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    service = make_service(tmp_path, {'account-1': 'http://127.0.0.1:9223'})
    monkeypatch.setattr(service, '_start_account_worker', lambda _account: None)
    assert service._startup_accounts == set()
    service.start()
    assert service.account_urls['account-1'] == 'http://127.0.0.1:19460'
    assert f'--user-data-dir={profile}' in launch.call_args.args[0]
    service.stop()
    restored = make_service(tmp_path, {'account-1': 'http://127.0.0.1:9223'})
    monkeypatch.setattr(restored, '_start_account_worker', lambda _account: None)
    restored.start()
    assert restored.account_urls['account-1'] == 'http://127.0.0.1:19460'
    assert restored._startup_accounts == set()
    restored.stop()


def test_managed_restart_reuses_profile_and_stops_owned_browser(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import magpie_wan.service as module
    browser = tmp_path / 'chrome'
    browser.touch()
    monkeypatch.setattr(module, '_browser_executable', lambda: browser)
    ports = iter([19457, 19458])
    monkeypatch.setattr(module, '_free_cdp_port', lambda **kwargs: next(ports))
    processes = [Mock(), Mock()]
    for process in processes:
        process.poll.return_value = None
    launch = Mock(side_effect=processes)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    service = make_service(tmp_path)
    monkeypatch.setattr(service, '_start_account_worker', lambda _: None)
    account = service.launch_browser_account()['id']
    profile = service.store.data_dir / 'browser-profiles' / account
    service.stop()
    processes[0].terminate.assert_called_once()
    assert profile.exists()
    restored = make_service(tmp_path)
    monkeypatch.setattr(restored, '_start_account_worker', lambda _: None)
    restored.start()
    assert restored.account_urls[account].endswith(':19458')
    assert f'--user-data-dir={profile}' in launch.call_args.args[0]
    restored.stop()
    processes[1].terminate.assert_called_once()


def test_identity_lookup_preserves_unique_binding_and_job_owner(tmp_path):
    service = make_service(tmp_path, {'old': 'http://127.0.0.1:9223', 'new': 'http://127.0.0.1:9224'})
    service.store.verify_identity('old', 'Id:123')
    assert service.store.account_for_identity('Id:123') == 'old'
    assert service.store.account_for_identity('Id:456') is None
    with pytest.raises(ServiceError, match='already bound'):
        service.store.verify_identity('new', 'Id:123')
    service.stop()


def test_balance_routing_filters_sessions_and_rechecks_latest_wallet(tmp_path):
    import time
    service = make_service(tmp_path, {a: f'http://localhost:{9300+i}' for i, a in enumerate('abcde')})
    frame = make_frame(tmp_path / 'frame.png')
    body = {'prompt': 'move', 'first_frame': str(frame), 'routing': 'balanced'}
    for a, balance in zip('abcde', (10, 80, 500, 900, None)):
        service.account_profiles[a].update(status='connected', balance=balance, checked_at=time.time())
    service.account_profiles['c']['checked_at'] -= 121
    service.account_profiles['d']['status'] = 'blocked'
    first, _ = service.submit({**body, 'idempotency_key': 'balanced'})
    assert first['account'] == 'b'
    service.account_profiles['b']['balance'] = 5
    assert service.submit(body)[0]['account'] == 'a'
    assert service.submit({**body, 'idempotency_key': 'balanced'}) == (first, False)
    with pytest.raises(ServiceError, match='different input'):
        service.submit({**body, 'idempotency_key': 'balanced', 'routing': 'preferred'})
    assert service.submit({**body, 'routing': 'preferred', 'preferred_account': 'b'})[0]['account'] == 'b'
    with pytest.raises(ServiceError, match='首选账号'):
        service.submit({**body, 'routing': 'preferred', 'preferred_account': 'c'})
    assert service.submit({**body, 'account': 'c'})[0]['account'] == 'c'
    for a in 'ab':
        service.account_profiles[a]['balance'] = 0
    before = set(service.store.jobs_dir.iterdir())
    with pytest.raises(ServiceError, match='没有余额'):
        service.submit(body)
    assert set(service.store.jobs_dir.iterdir()) == before
    service.stop()


def test_concurrent_balance_ties_use_queue_and_idempotency(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    service = make_service(tmp_path, {'a': 'http://localhost:9301', 'b': 'http://localhost:9302'})
    for p in service.account_profiles.values():
        p.update(status='connected', balance=50, checked_at=time.time())
    frame = make_frame(tmp_path / 'frame.png')
    body = {'prompt': 'move', 'first_frame': str(frame), 'routing': 'balanced'}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: service.submit(body)[0], range(12)))
        repeats = list(pool.map(lambda _: service.submit({**body, 'idempotency_key': 'same'}), range(8)))
    assert sum(j['account'] == 'a' for j in jobs) == 6
    assert sum(j['account'] == 'b' for j in jobs) == 6
    assert len({j['id'] for j, _ in repeats}) == 1
    assert sum(created for _, created in repeats) == 1
    service.stop()


def test_clean_device_starts_without_phantom_accounts_and_restores_login(tmp_path, monkeypatch):
    import magpie_wan.service as module
    from unittest.mock import Mock
    monkeypatch.delenv('WAN_MODELSCOPE_ACCOUNTS', raising=False)
    monkeypatch.delenv('WAN_MODELSCOPE_CDP_URLS', raising=False)
    assert parse_accounts() == {}
    service = WanService(data_dir=tmp_path / 'data', accounts={}, media_roots=[tmp_path])
    service.start()
    assert service.accounts == []
    browser = tmp_path / 'chrome'
    browser.touch()
    monkeypatch.setattr(module, '_browser_executable', lambda: browser)
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, 'Popen', Mock(return_value=process))
    ports = iter([19460, 19461])
    monkeypatch.setattr(module, '_free_cdp_port', lambda **kwargs: next(ports))
    monkeypatch.setattr(service, '_start_account_worker', lambda _: None)
    account = service.launch_browser_account(pending=True)['id']
    assert account == 'account-1'
    service.store.verify_identity(account, 'Name:original')
    service.activate_browser_account(account)
    service.stop()
    restored = WanService(data_dir=tmp_path / 'data', accounts={}, media_roots=[tmp_path])
    monkeypatch.setattr(restored, '_start_account_worker', lambda _: None)
    restored.start()
    assert restored.accounts == ['account-1']
    assert restored.account_urls[account].endswith(':19461')
    restored.store.verify_identity(account, 'Name:original')
    with pytest.raises(ServiceError, match='identity mismatch'):
        restored.store.verify_identity(account, 'Name:other')
    restored.stop()


def test_blockage_preserves_provider_history_and_separates_timing(tmp_path):
    import time
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / 'frame.png')
    first = service.submit({'prompt': 'one', 'first_frame': str(frame)})[0]
    second = service.submit({'prompt': 'two', 'first_frame': str(frame)})[0]
    now = time.time()
    service.store.update(first['id'], status='in_progress', provider_task_id='remote', provider_status='RUNNING',
                         submitted_at=now-300, started_at=now-310, last_polled_at=now-200)
    service.account_profiles['account-1'].update(status='unavailable', checked_at=now)
    service._account_unavailable_since['account-1'] = now-100
    public = service.job_details(service.store.get(first['id']))
    assert public['status'] == 'in_progress'
    assert public['last_confirmed_provider_status'] == 'RUNNING'
    assert '账号连接' in public['progress_message']
    assert public['timing']['connection_unavailable_seconds'] >= 100
    assert public['timing']['generation_seconds'] >= 300
    assert public['blockage']['waiting_job_ids'] == [second['id']]
    assert service.public_job(second)['blockage']['blocked_by_job_id'] == first['id']
    assert public['quality_acceptance'] == 'not_evaluated'
    assert any(e['kind'] == 'in_progress' for e in public['events'])
    service.stop()


def test_local_queue_controls_preserve_submitted_jobs_and_audit_reassignment(tmp_path):
    import time
    service = make_service(tmp_path, {'a': 'http://localhost:9301', 'b': 'http://localhost:9302'})
    frame = make_frame(tmp_path / 'frame.png')
    job = service.submit({'prompt': 'one', 'first_frame': str(frame), 'account': 'a'})[0]
    assert service.control_job(job['id'], {'action': 'pause'})['status'] == 'paused'
    assert service.store.next_for_account('a') is None
    service.account_profiles['b'].update(status='connected', checked_at=time.time())
    assert service.control_job(job['id'], {'action': 'reassign', 'account': 'b'})['account'] == 'b'
    assert service.control_job(job['id'], {'action': 'resume'})['status'] == 'queued'
    assert service.store.next_for_account('b')['id'] == job['id']
    service._account_operations['b'].acquire()
    try:
        with pytest.raises(ServiceError, match='正在操作'):
            service.control_job(job['id'], {'action': 'cancel'})
    finally:
        service._account_operations['b'].release()
    assert service.control_job(job['id'], {'action': 'cancel'})['status'] == 'cancelled'
    assert service.store.active_counts(['b'])['b'] == 0
    assert service.public_job(service.store.get(job['id']))['poll_after_seconds'] is None
    events = service.store.events(job['id'])
    assert any(e.get('previous_account') == 'a' and e.get('account') == 'b' for e in events)
    for status in ('uploading', 'submitting', 'unknown_submission', 'in_progress', 'download_pending', 'completed'):
        service.store.update(job['id'], status=status, provider_task_id='existing' if status != 'unknown_submission' else None)
        for action in ('cancel', 'pause', 'reassign', 'resume'):
            with pytest.raises(ServiceError, match='尚未提交'):
                service.control_job(job['id'], {'action': action, 'account': 'a'})
    service.stop()


def test_failure_details_are_stage_specific_and_do_not_dump_private_responses(tmp_path):
    from unittest.mock import Mock
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / 'frame.png')
    def job():
        return service.submit({'prompt': 'one', 'first_frame': str(frame)})[0]
    client = Mock()
    current = job()
    client.upload.side_effect = ValueError('private /Users/example path cookie=secret')
    service._submit_job(client, current)
    public = service.job_details(service.store.get(current['id']))
    assert public['failure']['stage'] == 'upload'
    assert public['recovery']['safe_to_resubmit']
    assert 'private' not in str(public['failure'])
    client.upload.side_effect = None
    client.estimate.side_effect = provider.ModelScopeAPIError('/quote', 503, 'UNAVAILABLE')
    current = job()
    service._submit_job(client, current)
    failure = service.public_job(service.store.get(current['id']))['failure']
    assert failure['stage'] == 'estimate' and failure['http_status'] == 503
    client.estimate.side_effect = None
    client.estimate.return_value = 10
    client.submit.side_effect = TimeoutError('token=private')
    current = job()
    service._submit_job(client, current)
    public = service.public_job(service.store.get(current['id']))
    assert public['internal_status'] == 'unknown_submission'
    assert not public['recovery']['safe_to_resubmit']
    assert public['recovery']['action'] == 'reconcile'
    for body in ({'provider_task_id': 'task', 'confirm_no_task': True}, {}, {'provider_task_id': 'https://secret'}):
        with pytest.raises(ServiceError):
            service.control_job(current['id'], {'action': 'resolve', **body})
    linked = service.control_job(current['id'], {'action': 'resolve', 'provider_task_id': 'task-123'})
    assert linked['provider_task_id'] == 'task-123'
    client.status.return_value = {'status': 'FAILED', 'errorCode': 'BAD_INPUT',
        'errorMessage': 'invalid token=secret https://secret.url/path?password=123',
        'cookies': 'NEVER_COPY_THIS_FIELD'}
    service._poll_job(client, linked)
    public = service.job_details(service.store.get(current['id']))
    assert public['failure']['provider_code'] == 'BAD_INPUT'
    assert 'secret' not in str(public['failure'])
    assert 'NEVER_COPY' not in str(public)
    assert not public['recovery']['safe_to_resubmit']
    assert public['recovery']['actual_charge'] is None
    client.status.return_value = {'status': 'FAILED'}
    service._poll_job(client, linked)
    assert '服务端未提供原因' in service.store.get(current['id'])['error']
    service.stop()


def test_models_diagnostics_and_control_http_authentication(tmp_path):
    import threading
    import requests
    from http.server import ThreadingHTTPServer
    from magpie_wan.service import make_handler, PROTOCOL_VERSION
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / 'frame.png')
    job = service.submit({'prompt': 'one', 'first_frame': str(frame)})[0]
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(service, 'private'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    headers = {'Authorization': 'Bearer private'}
    try:
        assert requests.get(base+'/v1/diagnostics', timeout=3).status_code == 401
        for route in ('/v1/videos/estimate', '/v1/models/search'):
            assert requests.post(base+route, json={}, timeout=3).status_code == 401
        assert requests.post(base+f'/v1/videos/{job["id"]}/control', json={'action': 'cancel'}, timeout=3).status_code == 401
        health = requests.get(base+'/v1/diagnostics', headers=headers, timeout=3).json()
        assert health['protocol_version'] == PROTOCOL_VERSION
        assert health['affected_jobs'][0]['id'] == job['id']
        models = requests.get(base+'/v1/models', headers=headers, timeout=3).json()['data']
        assert len(models) == 9
        assert models[2]['id'] == 'wan2.2-ti2v-5b'
        assert not models[2]['last_frame_supported']
        assert models[2]['provider_verified_at'] is None
        assert models[2]['quality_acceptance'] == 'not_evaluated'
        assert models[3]['id'] == 'minimax-h3'
        assert models[3]['last_frame_supported']
        result = requests.post(base+f'/v1/videos/{job["id"]}/control', headers=headers,
                               json={'action': 'pause'}, timeout=3)
        assert result.status_code == 200 and result.json()['status'] == 'paused'
    finally:
        server.shutdown(); thread.join(timeout=3); server.server_close(); service.stop()

@pytest.mark.parametrize('model', ['minimax-h3', 'ltx-2.3', 'wan2.2-ti2v-5b', 'wan2.2-t2v-a14b', 'wan2.1-t2v-14b', 'wan2.1-t2v-1.3b'])
def test_text_to_video_without_frames_survives_restart(tmp_path, model):
    import json
    service = make_service(tmp_path)
    job, _ = service.submit({'mode': 't2v', 'model': model, 'prompt': 'a ball rolls', 'size': '1280x720'})
    assert job['first_frame'] == '' and job['last_frame'] is None
    assert json.loads(job['generation_params'])['mode'] == 't2v'
    assert service.public_job(job)['input_frames'] == []
    service.stop()
    restarted = make_service(tmp_path)
    assert restarted.store.get(job['id'])['model'] == model
    restarted.stop()


def test_parameters_persist_affect_idempotency_and_reach_provider(tmp_path):
    import json
    from unittest.mock import Mock
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / 'first.png')
    body = {'first_frame': str(frame), 'model': 'ltx-2.3', 'prompt': 'dance', 'idempotency_key': 'params',
            'accelerate_lora': False, 'num_inference_steps': 25, 'guidance_scale': 4,
            'checkpoint_version_id': 1234, 'vae_version_id': 5678, 'add_watermark': True,
            'loras': [{'model_version_id': 999, 'scale': .75, 'trigger_words': ['motion']}]}
    job, _ = service.submit(body)
    assert service.submit(body)[1] is False
    with pytest.raises(ServiceError, match='different input'):
        service.submit({**body, 'num_inference_steps': 26})
    client = Mock()
    client.estimate.return_value = 10
    client.submit.return_value = 'task'
    service._submit_job(client, job)
    for name in ('loras', 'guidance_scale', 'num_inference_steps', 'checkpoint_version_id', 'vae_version_id', 'add_watermark'):
        assert client.request.call_args.kwargs[name] == body[name]
    assert service.public_job(job)['generation_params']['loras'] == body['loras']
    # Legacy queued records retain the old no-acceleration call shape.
    service.store.update(job['id'], generation_params=None)
    service._submit_job(client, service.store.get(job['id']))
    assert 'accelerate_lora' not in client.request.call_args.kwargs
    service.stop()


@pytest.mark.parametrize('params', [
    {'accelerate_lora': 'yes'}, {'accelerate_lora': True, 'num_inference_steps': 20},
    {'guidance_scale': float('nan')}, {'num_inference_steps': 51}, {'vae_version_id': True},
    {'loras': [{'model_version_id': 1, 'scale': float('inf')}]},
    {'loras': [{'model_version_id': 1}, {'model_version_id': 1}]},
    {'loras': [{'model_version_id': 1, 'unexpected': 'field'}]},
])
def test_invalid_generation_parameters_are_rejected_before_queue(tmp_path, params):
    service = make_service(tmp_path)
    with pytest.raises(ServiceError):
        service.submit({'mode': 't2v', 'model': 'minimax-h3', 'prompt': 'motion', **params})
    assert service.store.list_jobs() == []
    service.stop()


def test_quote_validation_creates_no_job_or_upload(tmp_path):
    service = make_service(tmp_path)
    spec = service._submit({'mode': 't2v', 'model': 'minimax-h3', 'prompt': 'motion'}, quote_only=True)
    assert spec['generation_params']['accelerate_lora'] is True
    assert service.store.list_jobs() == []
    assert not list(service.store.jobs_dir.iterdir())
    service.stop()


def test_old_idempotency_keys_return_original_jobs_after_upgrade(tmp_path):
    from magpie_wan.service import _fingerprint
    service = make_service(tmp_path)
    frame = make_frame(tmp_path / 'first.png')
    body = {'prompt': 'move', 'first_frame': str(frame), 'idempotency_key': 'legacy'}
    job, _ = service.submit(body)
    legacy_hash = _fingerprint(frame, None, 'move', job['model'], job['seconds'], job['width'], job['height'], None, None, -1)
    service.store.update(job['id'], input_fingerprint=legacy_hash, generation_params=None)
    retried, created = service.submit(body)
    assert retried['id'] == job['id'] and not created
    with pytest.raises(ServiceError, match='different input'):
        service.submit({**body, 'accelerate_lora': True})
    service.stop()
