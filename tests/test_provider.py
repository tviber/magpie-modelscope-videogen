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


def test_rejected_submission_without_reason_does_not_dump_response():
    client = provider.MS.__new__(provider.MS)
    client.post = Mock(return_value={'Data': {'success': False, 'private_cookie': 'never-save-this'}})
    with pytest.raises(provider.SubmissionRejected) as error:
        client.submit({})
    assert str(error.value) == ''


@pytest.mark.parametrize("end_url", [None, "last-url"])
def test_minimax_uses_live_version_and_sampling_defaults(end_url):
    client = provider.MS.__new__(provider.MS)
    client.default_size = None
    client.models = {"MINIMAX_H3": {
        "versionId": 982725,
        "defaultParams": {"basicDiffusionArgs": {"numInferenceSteps": 20}},
    }}
    request = client.request("MINIMAX_H3", "first-url", end_url, "roll slowly", 3,
                             width=1280, height=720)
    assert request["modelArgs"]["checkpointModelVersionId"] == 982725
    assert request["basicDiffusionArgs"] == {
        "width": 1280, "height": 720, "totalSeconds": 3, "seed": -1, "numInferenceSteps": 20}
    assert request["promptArgs"] == {"prompt": "roll slowly"}
    assert request["videoInferenceArgs"] == {"accelerateLora": False}
    assert request["imageInputFrontArgs"]["image"] == "first-url"
    assert request.get("extraImageInputFrontArgs") == ({"image": end_url, "stength": 1} if end_url else None)
    client.models["MINIMAX_H3"]["versionId"] = 999999
    client.models["MINIMAX_H3"]["defaultParams"]["basicDiffusionArgs"]["numInferenceSteps"] = 24
    refreshed = client.request("MINIMAX_H3", "first-url", end_url, "roll slowly", 3,
                               width=1280, height=720)
    assert refreshed["modelArgs"]["checkpointModelVersionId"] == 999999
    assert refreshed["basicDiffusionArgs"]["numInferenceSteps"] == 24


def test_full_ltx_text_request_and_trigger_words():
    client = provider.MS.__new__(provider.MS)
    client.default_size = None
    client.models = {'LTX_VIDEO_2_3': {'versionId': 701280, 'defaultParams': {'basicDiffusionArgs': {'guidanceScale': 4, 'numInferenceSteps': 50}}}}
    request = client.request('LTX_VIDEO_2_3', None, None, 'hello', 3, width=1280, height=768,
                             mode='t2v', accelerate_lora=True, vae_version_id=77, checkpoint_version_id=88,
                             loras=[{'model_version_id': 99, 'scale': .6, 'trigger_words': ['style','style']}], add_watermark=True)
    assert request['predictType'] == 'TXT_2_VIDEO'
    assert 'imageInputFrontArgs' not in request
    assert request['promptArgs']['prompt'] == 'hello,style'
    assert request['modelArgs'] == {'checkpointModelVersionId': 88, 'vaeModelVersionId': 77,
                                    'loraArgs': [{'modelVersionId': 99, 'scale': .6}]}
    assert request['basicDiffusionArgs']['numInferenceSteps'] == 4
    assert request['basicDiffusionArgs']['guidanceScale'] == 4
    client.post = Mock(return_value={'Data': {'success': True, 'data': {'taskId': 'task'}}})
    assert client.submit(request) == 'task'
    assert client.post.call_args.args[1]['addWaterMark'] is True


def test_catalog_returns_only_usable_version_metadata():
    client = provider.MS.__new__(provider.MS)
    client._check_identity = Mock()
    client._api = Mock(return_value={'Data': {'Model': {'TotalCount': 1, 'Models': [{
        'MuseInfo': {'model': {'modelName': 'owner/style'}, 'versions': [{
            'id': 42, 'stableDiffusionSubVersion': 'MINIMAX_H3',
            'modelVersion': {'showName': 'v1', 'modelUrl': 'modelscope://owner/style?revision=v1',
                             'triggerWords': '["style"]'}}]}, 'private': 'not-returned'}]}}})
    result = client.catalog('MINIMAX_H3', query='style')
    assert result['data'][0]['versions'][0]['model_version_id'] == 42
    assert result['data'][0]['versions'][0]['trigger_words'] == ['style']
    assert 'private' not in str(result)
    criteria = client._api.call_args.kwargs['data']['Criterion']
    assert criteria[0]['values'] == ['MINIMAX_H3']
