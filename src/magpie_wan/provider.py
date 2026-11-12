"""ModelScope's browser-session adapter for WAN video generation.

This module only speaks to ModelScope. It does not own project shot names,
artifact manifests, acceptance state, output layout, or prompt preferences.
"""

import io
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from PIL import Image


class SubmissionRejected(RuntimeError):
    """ModelScope explicitly declined to create a task."""


MODELSCOPE_HOSTS = {"modelscope.cn", "www.modelscope.cn"}


class ModelScopeAPIError(RuntimeError):
    """An API rejection, with a safe, user-facing explanation and endpoint name."""

    def __init__(self, endpoint, status, code=None):
        self.endpoint = endpoint
        self.status = status
        self.code = code
        if status == 401 and endpoint == "/api/v1/users/login/info":
            message = ("ModelScope rejected this browser session (HTTP 401). The profile may show a cached username; "
                       "reload this browser profile and sign in again if ModelScope asks.")
        elif status == 401 and endpoint == "/openapi/v1/magicubes/balance":
            message = ("ModelScope rejected the magicube balance request (HTTP 401). "
                       "The browser login session may need to be refreshed.")
        else:
            message = f"ModelScope request {endpoint} failed (HTTP {status})"
            if code:
                message += f" [{code}]"
        super().__init__(message)


def trusted_url(url, *, upload=False):
    """Accept only official HTTPS origins; OSS uploads may not redirect."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        allowed = host in MODELSCOPE_HOSTS
        if upload:
            allowed = allowed or bool(re.fullmatch(
                r"[a-z0-9][a-z0-9-]*\.oss-[a-z0-9-]+\.aliyuncs\.com", host))
        return (parsed.scheme == "https" and allowed and parsed.port in (None, 443)
                and parsed.username is None and parsed.password is None
                and not any(char.isspace() or char == "\\" for char in url))
    except (ValueError, TypeError):
        return False


def login_identity(payload):
    data = payload.get("Data") or {}
    # Prefer the provider's immutable ID. Names are used only by older responses;
    # a later change of identity format fails closed and requires investigation.
    for field in ("Id", "UserId", "Username", "Name"):
        value = data.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value:
            return f"{field}:{value}"
    raise RuntimeError("No signed-in ModelScope account found in this browser session")


def account_label(index):
    """Return a generic label without exposing a local debugging endpoint."""
    return f"account-{int(index) + 1}"


def find_url(data):
    """Find a video URL in ModelScope's nested task response."""
    if isinstance(data, str):
        return data if data.startswith("http") and (".mp4" in data or "video" in data) else None
    if isinstance(data, dict):
        for key in ("videoUrl", "url", "video_url", "videoURL"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in data.values():
            url = find_url(value)
            if url:
                return url
    if isinstance(data, list):
        for value in data:
            url = find_url(value)
            if url:
                return url
    return None


class MS:
    def __init__(self, page, cdp, default_size=None):
        self.pg = page
        self.cdp = cdp
        self.default_size = default_size
        self.expected_identity = self.identity()
        template = self.get("/api/v1/muse/predict/defaultTemplateV2")
        self.models = {
            model["sub_value"]: model
            for model in template["Data"]["data"]["VIDEO"]["supportSDVersionList"]
        }

    @staticmethod
    def _size(width, height, default_size):
        if width is None and height is None:
            if default_size is None:
                raise ValueError("width and height are required for this ModelScope request")
            return default_size
        if width is None or height is None:
            raise ValueError("width and height must be supplied together")
        return int(width), int(height)

    def get(self, path):
        self._check_identity()
        return self._api("get", path)

    def post(self, path, body):
        self._check_identity()
        return self._api("post", path, data=body)

    def _api(self, method, path, **kwargs):
        if not trusted_url(self.pg.url) or not path.startswith("/") or path.startswith("//"):
            raise RuntimeError("A trusted ModelScope HTTPS page is required")
        origin = "https://" + urlsplit(self.pg.url).hostname
        # Use the browser context's HTTP client and cookies, not page JavaScript.
        # Absolute URLs and disabled redirects prevent navigation from rerouting data.
        response = getattr(self.pg.request, method)(
            origin + path, max_redirects=0, timeout=30_000, **kwargs)
        try:
            if not response.ok:
                code = None
                try:
                    payload = response.json()
                    code = payload.get("Code") or payload.get("code")
                except Exception:
                    pass
                raise ModelScopeAPIError(path, response.status, code)
            return response.json()
        finally:
            response.dispose()

    def identity(self):
        return login_identity(self._api("get", "/api/v1/users/login/info"))

    def profile(self):
        """Return a user-facing label without exposing cookies or CDP details."""
        payload = self._api("get", "/api/v1/users/login/info")
        if login_identity(payload) != self.expected_identity:
            raise RuntimeError("ModelScope login identity changed; reconnect the original account")
        data = payload.get("Data") or {}
        for field in ("Email", "Username", "NickName", "Nickname", "Name"):
            value = data.get(field)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()
        for field in ("Email", "Username", "NickName", "Nickname", "Name"):
            value = (data.get("UserInfo") or {}).get(field) if isinstance(data.get("UserInfo"), dict) else None
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()
        return "ModelScope"

    def _check_identity(self):
        if self.identity() != self.expected_identity:
            raise RuntimeError("ModelScope login identity changed; reconnect the original account")

    def balance(self):
        return self.get("/openapi/v1/magicubes/balance").get("data")

    def me(self):
        """Return the signed-in identity, or None when the session is not logged in."""
        try:
            return self.identity()
        except Exception:
            return None

    def upload(self, path, width=None, height=None):
        width, height = self._size(width, height, self.default_size)
        image = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        upload_url = self.post("/api/v1/rm/uploadUrl", {
            "FileName": Path(path).stem + ".jpg",
            "Type": "AIGC_COVER_IMAGE",
        })["Data"]["UploadUrl"]
        if not trusted_url(upload_url, upload=True):
            raise RuntimeError("ModelScope returned an untrusted upload URL")
        response = requests.put(upload_url, data=buffer.getvalue(),
                     headers={"Content-Type": "application/octet-stream", "x-oss-meta-author": "aliy"},
                     timeout=120, allow_redirects=False)
        try:
            if not 200 <= response.status_code < 300:
                raise RuntimeError("ModelScope image upload failed")
        finally:
            response.close()
        return self.post("/api/v1/rm/downloadUrl", {
            "FileUrl": upload_url.split("?")[0],
            "Type": "AIGC_COVER_IMAGE",
        })["Data"]["DownloadUrl"]

    def request(self, model, start_url, end_url, prompt, seconds, *, width=None, height=None,
                style=None, negative_prompt=None, seed=-1):
        width, height = self._size(width, height, self.default_size)
        prompt_args = {"prompt": prompt}
        if style:
            prompt_args["prompt"] = f"{prompt}. {style}"
        if negative_prompt is not None:
            prompt_args["negativePrompt"] = negative_prompt
        request = {
            "predictType": "IMG_2_VIDEO",
            "promptArgs": prompt_args,
            "basicDiffusionArgs": {"width": width, "height": height, "totalSeconds": seconds, "seed": seed},
            "modelArgs": {"checkpointModelVersionId": self.models[model]["versionId"], "loraArgs": []},
            "videoInferenceArgs": {"accelerateLora": False},
            "imageInputFrontArgs": {"image": start_url, "stength": 1},
        }
        if end_url:
            request["extraImageInputFrontArgs"] = {"image": end_url, "stength": 1}
        return request

    def estimate(self, request):
        response = self.post("/api/v1/muse/predict/getEstimateConsumeComputeQuota", {
            "type": "VIDEO",
            "videoPredictTaskRequest": request,
        })
        return (response.get("Data") or {}).get("data")

    def submit(self, request):
        body = dict(request, addWaterMark=False)
        for _ in range(60):
            response = self.post("/api/v1/muse/predict/task/video/submit", body)
            data = response.get("Data") or {}
            if data.get("success") and (data.get("data") or {}).get("taskId"):
                return data["data"]["taskId"]
            # A missing reason must not turn the whole response (which can
            # contain private URLs or session data) into a persisted exception.
            message = data.get("message") or response.get("Message")
            message = message if isinstance(message, str) else ""
            if "正在" in message or "进行中" in message or "排队" in message:
                time.sleep(30)
                continue
            raise SubmissionRejected(message)
        raise RuntimeError("submit retries exhausted")

    def status(self, task_id):
        response = self.get(f"/api/v1/muse/predict/task/video/status?taskId={quote(str(task_id), safe='')}")
        return (response.get("Data") or {}).get("data") or {}

    def wait(self, task_id, timeout=2700):
        started, last_status = time.time(), None
        while time.time() - started < timeout:
            status = self.status(task_id).get("status")
            if status != last_status:
                print(f"  [{time.time() - started:4.0f}s] {status}", flush=True)
                last_status = status
            if status in ("SUCCEED", "FAILED", "CANCELED"):
                return self.status(task_id)
            time.sleep(15)
        raise TimeoutError(task_id)


def connect(playwright, cdp, default_size=None):
    browser = playwright.chromium.connect_over_cdp(cdp)
    pages = [page for context in browser.contexts for page in context.pages if trusted_url(page.url)]
    if not pages:
        raise RuntimeError("Open an official ModelScope HTTPS page in the signed-in browser session")
    failures = []
    for page in pages:
        try:
            return MS(page, cdp, default_size=default_size)
        except Exception as exc:
            if isinstance(exc, ModelScopeAPIError):
                failures.append(str(exc))
            else:
                failures.append(f"ModelScope page could not initialize ({type(exc).__name__})")
            continue
    detail = "; ".join(dict.fromkeys(failures))
    raise RuntimeError(f"No usable ModelScope session was found. {detail}")
