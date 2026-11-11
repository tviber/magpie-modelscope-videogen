from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from magpie_wan import provider


@pytest.mark.parametrize("url", [
    "https://www.google.com/search?q=modelscope",
    "https://example.org/?modelscope",
    "https://modelscope.cn.evil.org/",
    "http://modelscope.cn/", "https://modelscope.cn:444/",
    "https://modelscope.cn@evil.org/", "https://user@modelscope.cn/",
    "https://modelscope.cn\\@evil.org/", "file:///modelscope.cn",
])
def test_untrusted_tabs_never_make_api_calls(url):
    page = SimpleNamespace(url=url, request=Mock())
    playwright = SimpleNamespace(chromium=Mock())
    playwright.chromium.connect_over_cdp.return_value.contexts = [SimpleNamespace(pages=[page])]
    with pytest.raises(RuntimeError, match="official ModelScope"):
        provider.connect(playwright, "unused")
    assert not page.request.mock_calls


def api_response(data):
    response = Mock(ok=True)
    response.json.return_value = data
    return response


def page_with_identity(identity):
    page = SimpleNamespace(url="https://modelscope.cn/studios", request=Mock())
    def get(url, **kwargs):
        if url.endswith("/login/info"):
            return api_response({"Data": {"Id": identity}})
        return api_response({"Data": {"data": {"VIDEO": {"supportSDVersionList": []}}}})
    page.request.get.side_effect = get
    return page


def test_connection_skips_fake_and_logged_out_tabs():
    fake = SimpleNamespace(url="https://example.org/?modelscope", request=Mock())
    logged_out = page_with_identity(None)
    valid = page_with_identity(123)
    playwright = SimpleNamespace(chromium=Mock())
    playwright.chromium.connect_over_cdp.return_value.contexts = [
        SimpleNamespace(pages=[fake, logged_out]), SimpleNamespace(pages=[valid])]
    client = provider.connect(playwright, "unused")
    assert client.pg is valid
    assert not fake.request.mock_calls
    for call in valid.request.get.call_args_list:
        assert call.args[0].startswith("https://modelscope.cn/")
        assert call.kwargs["max_redirects"] == 0


def test_navigation_and_changed_login_block_mutations():
    page = page_with_identity(123)
    client = provider.MS(page, "unused")
    page.url = "https://example.org/?modelscope"
    with pytest.raises(RuntimeError, match="trusted"):
        client.post("/api/submit", {"prompt": "private"})
    page.url = "https://modelscope.cn/"
    page.request.get.side_effect = lambda *a, **k: api_response({"Data": {"Id": 456}})
    with pytest.raises(RuntimeError, match="identity changed"):
        client.post("/api/submit", {"prompt": "private"})
    page.request.post.assert_not_called()


def test_api_401_preserves_endpoint_and_login_guidance():
    page = page_with_identity(123)

    def unauthorized(*_args, **_kwargs):
        response = Mock(ok=False, status=401)
        response.json.return_value = {"Code": "Unauthorized"}
        response.dispose = Mock()
        return response

    page.request.get.side_effect = unauthorized
    client = provider.MS.__new__(provider.MS)
    client.pg = page
    with pytest.raises(provider.ModelScopeAPIError, match="browser session") as error:
        client.identity()
    assert error.value.endpoint == "/api/v1/users/login/info"
    assert error.value.status == 401
    assert error.value.code == "Unauthorized"


@pytest.mark.parametrize("url", ["https://evil.org/image", "http://bucket.oss-cn-hangzhou.aliyuncs.com/a",
                                    "https://bucket.oss-cn-hangzhou.aliyuncs.com.evil.org/a"])
def test_upload_rejects_untrusted_destination(tmp_path, monkeypatch, url):
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16)).save(frame)
    client = provider.MS(page_with_identity(123), "unused")
    client.post = Mock(return_value={"Data": {"UploadUrl": url}})
    put = Mock()
    monkeypatch.setattr(provider.requests, "put", put)
    with pytest.raises(RuntimeError, match="untrusted upload"):
        client.upload(frame, 16, 16)
    put.assert_not_called()


def test_upload_does_not_follow_redirects(tmp_path, monkeypatch):
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16)).save(frame)
    client = provider.MS(page_with_identity(123), "unused")
    client.post = Mock(return_value={"Data": {
        "UploadUrl": "https://bucket.oss-cn-hangzhou.aliyuncs.com/frame?signature=test"}})
    put = Mock(return_value=Mock(status_code=307))
    monkeypatch.setattr(provider.requests, "put", put)
    with pytest.raises(RuntimeError, match="upload failed"):
        client.upload(frame, 16, 16)
    assert put.call_args.kwargs["allow_redirects"] is False
    assert client.post.call_count == 1
