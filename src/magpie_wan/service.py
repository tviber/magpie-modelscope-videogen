"""Durable local HTTP job service for the ModelScope WAN browser adapter.

The service owns task state and account queues. Magpie tools submit jobs and
return immediately; ModelScope work continues in one worker per CDP account.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import socket
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager

if os.name == "nt":
    import msvcrt
else:
    import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import Image

from . import provider as ms_provider
from . import __version__


DEFAULT_MODEL = "wan2.1-flf2v-14b"
MODEL_IDS = {
    "wan2.1-flf2v-14b": "WAN_VIDEO_2_1_FLF2V_14_B",
    "wan2.2-i2v-a14b": "WAN_VIDEO_2_2_I2V_A_14_B",
    "wan2.2-ti2v-5b": "WAN_VIDEO_2_2_TI2V_5_B",
    "minimax-h3": "MINIMAX_H3",
    "ltx-2.3": "LTX_VIDEO_2_3",
    "wan2.2-t2v-a14b": "WAN_VIDEO_2_2_T2V_A_14_B",
    "wan2.1-t2v-1.3b": "WAN_VIDEO_2_1_T2V_1_3_B",
    "wan2.1-t2v-14b": "WAN_VIDEO_2_1_T2V_14_B",
    "wan2.1-i2v-14b": "WAN_VIDEO_2_1_I2V_14_B",
}
LAST_FRAME_MODELS = {DEFAULT_MODEL, "minimax-h3", "ltx-2.3"}
TEXT_MODELS = {"minimax-h3", "ltx-2.3", "wan2.2-ti2v-5b", "wan2.2-t2v-a14b",
               "wan2.1-t2v-1.3b", "wan2.1-t2v-14b"}
IMAGE_MODELS = set(MODEL_IDS) - {"wan2.2-t2v-a14b", "wan2.1-t2v-1.3b", "wan2.1-t2v-14b"}
MODEL_ALIASES = {**MODEL_IDS, **{value: value for value in MODEL_IDS.values()}}
MAX_BODY_BYTES = 64 * 1024
MAX_FRAME_BYTES = 30 * 1024 * 1024
MAX_PROMPT_CHARS = 20_000
MAX_STYLE_CHARS = 4_000
MAX_DURATION_SECONDS = 10
MAX_PROJECT_CHARS = 160
TERMINAL_PROVIDER_STATES = {"SUCCEED", "FAILED", "CANCELED"}
ACTIVE_STATES = ("queued", "uploading", "submitting", "in_progress", "download_pending", "unknown_submission", "paused")
PROTOCOL_VERSION = 3
PUBLIC_STATUS = {
    "queued": "queued",
    "uploading": "in_progress",
    "submitting": "in_progress",
    "in_progress": "in_progress",
    "download_pending": "in_progress",
    "unknown_submission": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "paused": "paused",
    "cancelled": "cancelled",
}


def safe_provider_failure(stage, exc=None, payload=None):
    """Keep only documented error fields, never entire responses or exception text."""
    payload = payload if isinstance(payload, dict) else {}
    reason = next((payload[k] for k in ("errorMessage", "failReason", "failureReason", "message")
                   if isinstance(payload.get(k), str) and payload[k].strip()), None)
    if isinstance(exc, ms_provider.SubmissionRejected):
        reason = str(exc)
    if reason:
        reason = re.sub(r"https?://[^\s]+", "[URL removed]", reason)
        reason = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]+", "[email removed]", reason)
        reason = re.sub(r"(?:[A-Za-z]:\\|/(?:Users|home|tmp|private)/)[^\s,;]+", "[path removed]", reason)
        reason = re.sub(r"(?i)\b(token|cookie|authorization|password|secret|session)\b\s*[:=]\s*[^\s,;]+",
                        r"\1=[redacted]", reason)
        reason = re.sub(r"\b(?:Bearer\s+)?[A-Za-z0-9_+=/-]{32,}\b", "[redacted]", reason)
        reason = reason[:400]
    code = payload.get("errorCode") or payload.get("code") or getattr(exc, "code", None)
    code = str(code) if isinstance(code, (str, int)) and re.fullmatch(r"[\w.-]{1,64}", str(code)) else None
    return {"stage": stage, "provider_code": code, "provider_reason": reason,
            "reason_available": bool(reason),
            "message": reason or "服务端未提供原因。",
            "http_status": getattr(exc, "status", None) if isinstance(exc, ms_provider.ModelScopeAPIError) else None,
            "exception_type": type(exc).__name__ if exc else None}


class ServiceError(ValueError):
    def __init__(self, message, status=400, code="invalid_request_error", details=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = details


def default_data_dir():
    if os.environ.get("WAN_SERVICE_DATA_DIR"):
        return Path(os.environ["WAN_SERVICE_DATA_DIR"]).expanduser()
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.environ["XDG_DATA_HOME"]).expanduser() / "magpie-wan"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "magpie-wan"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "magpie-wan"
    return Path.home() / ".local" / "share" / "magpie-wan"


def parse_accounts(raw=None):
    named = os.environ.get("WAN_MODELSCOPE_ACCOUNTS") if raw is None else None
    if named:
        try:
            accounts = json.loads(named)
        except json.JSONDecodeError as exc:
            raise ServiceError("WAN_MODELSCOPE_ACCOUNTS must be a JSON object of stable IDs to CDP URLs") from exc
        return {} if accounts == {} else validate_accounts(accounts)
    if raw is None and "WAN_MODELSCOPE_CDP_URLS" not in os.environ:
        return {}
    raw = raw if raw is not None else os.environ["WAN_MODELSCOPE_CDP_URLS"]
    urls = [item.strip() for item in raw.split(",") if item.strip()]
    if not urls:
        raise ServiceError("WAN_MODELSCOPE_CDP_URLS must contain at least one CDP URL")
    if len(set(urls)) != len(urls):
        raise ServiceError("WAN_MODELSCOPE_CDP_URLS contains duplicate URLs")
    return urls


def validate_accounts(accounts):
    if not isinstance(accounts, dict) or not accounts:
        raise ServiceError("Configure at least one stable account ID and CDP URL")
    for account, url in accounts.items():
        if not isinstance(account, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", account):
            raise ServiceError("Account IDs must contain only letters, numbers, underscores and hyphens")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ServiceError("Account CDP URLs must use HTTP or HTTPS")
    if len(set(accounts.values())) != len(accounts):
        raise ServiceError("Account CDP URLs must be unique")
    return accounts


def _browser_executable():
    candidates = []
    if sys.platform == "darwin":
        candidates.extend((
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ))
    elif os.name == "nt":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(variable)
            if base:
                candidates.append(Path(base) / "Google/Chrome/Application/chrome.exe")
                candidates.append(Path(base) / "Chromium/Application/chrome.exe")
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def _free_cdp_port(start=9223, excluded=()):
    excluded = {int(port) for port in excluded}
    for port in range(start, min(start + 1000, 65536)):
        if port in excluded:
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise ServiceError("No free local browser debugging port is available", 503, "browser_port_unavailable")


def parse_media_roots(raw=None):
    raw = raw if raw is not None else os.environ.get("WAN_MEDIA_ROOTS")
    if not raw:
        raise ServiceError("Set WAN_MEDIA_ROOTS to one or more directories containing input frames")
    roots = [Path(part.strip()).expanduser().resolve() for part in raw.split(os.pathsep) if part.strip()]
    if not roots or any(not path.is_dir() for path in roots):
        raise ServiceError("WAN_MEDIA_ROOTS must contain existing directories")
    return roots


def _parse_size(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ServiceError("size must be a string such as '1280x720'")
    match = re.fullmatch(r"\s*(\d{2,4})\s*[xX]\s*(\d{2,4})\s*", value)
    if not match:
        raise ServiceError("size must be written as WIDTHxHEIGHT, such as '1280x720'")
    width, height = map(int, match.groups())
    if not (256 <= width <= 1536 and 256 <= height <= 1536):
        raise ServiceError("width and height must each be between 256 and 1536")
    if width % 16 or height % 16:
        raise ServiceError("width and height must be multiples of 16")
    return width, height


def _infer_dimensions(image_path):
    with Image.open(image_path) as image:
        source_w, source_h = image.size
    source_long = max(source_w, source_h)
    max_source_edge = min(source_long, 1280)
    pairs = [(width, height) for width in range(256, 1281, 16) for height in range(256, 1281, 16)
             if abs((width / height) / (source_w / source_h) - 1) <= 0.015]
    if not pairs:
        raise ServiceError("frame aspect ratio cannot be represented by supported dimensions")
    at_or_below_source = [(width, height) for width, height in pairs
                          if max(width, height) <= max_source_edge]
    if at_or_below_source:
        width, height = max(at_or_below_source, key=lambda pair: (
            max(pair), -abs((pair[0] / pair[1]) / (source_w / source_h) - 1)))
        return width, height
    above_source = [(width, height) for width, height in pairs if max(width, height) > max_source_edge]
    width, height = min(above_source or pairs, key=lambda pair: (max(pair),
                                                                 abs((pair[0] / pair[1]) / (source_w / source_h) - 1)))
    return width, height


def _validate_frame(value, roots):
    if not isinstance(value, str) or not value:
        raise ServiceError("frame inputs must be absolute local file paths")
    source = Path(value).expanduser()
    if not source.is_absolute():
        raise ServiceError("frame inputs must be absolute local file paths")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ServiceError("frame file is unavailable") from exc
    if not resolved.is_file():
        raise ServiceError("frame input is not a file")
    if not any(resolved == root or root in resolved.parents for root in roots):
        allowed = [str(root) for root in roots]
        raise ServiceError(
            f"frame file is outside WAN_MEDIA_ROOTS. Allowed directories: {', '.join(allowed)}. "
            "Move the image there, or use copy_to_media_root=true in the Magpie submit tool.",
            code="frame_outside_media_roots", details={"allowed_roots": allowed})
    if resolved.stat().st_size > MAX_FRAME_BYTES:
        raise ServiceError("frame file is larger than 30 MiB")
    try:
        with Image.open(resolved) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise ServiceError("frame file is not a readable image") from exc
    if width < 16 or height < 16 or width > 16384 or height > 16384:
        raise ServiceError("frame image dimensions must be between 16 and 16384 pixels")
    return resolved, width, height


def _fingerprint(first_frame, last_frame, prompt, model, seconds, width, height, style, negative_prompt, seed,
                 project=None, shot=None, account=None, routing=None, preferred_account=None, generation_params=None):
    digest = hashlib.sha256()
    for image_path in (first_frame, last_frame):
        content = b"" if image_path is None else Path(image_path).read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    fields = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "width": width,
        "height": height,
        "style": style,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }
    # Keep legacy hashes unchanged for requests without grouping or account selection.
    for key, value in (("project", project), ("shot", shot), ("account", account),
                       ("routing", routing), ("preferred_account", preferred_account)):
        if value is not None:
            fields[key] = value
    if generation_params:
        fields["generation_params"] = generation_params
    digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    return "sha256:" + digest.hexdigest()


class JobStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.data_dir.chmod(0o700)
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.jobs_dir.chmod(0o700)
        self.db_path = self.data_dir / "jobs.sqlite3"
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def resolve_data_path(self, relative_path):
        if not isinstance(relative_path, str) or not relative_path:
            raise ServiceError("stored job path is invalid", 500, "server_error")
        portable = PurePosixPath(relative_path.replace("\\", "/"))
        if (portable.is_absolute() or PureWindowsPath(relative_path).drive
                or ".." in portable.parts):
            raise ServiceError("stored job path is invalid", 500, "server_error")
        path = Path(*portable.parts)
        resolved = (self.data_dir / path).resolve()
        if resolved == self.data_dir or self.data_dir not in resolved.parents:
            raise ServiceError("stored job path is invalid", 500, "server_error")
        return resolved

    def relative_data_path(self, path):
        return Path(path).resolve().relative_to(self.data_dir).as_posix()

    def _initialize(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                prompt TEXT NOT NULL,
                first_frame TEXT NOT NULL,
                last_frame TEXT,
                model TEXT NOT NULL,
                seconds INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                style TEXT,
                negative_prompt TEXT,
                seed INTEGER NOT NULL,
                input_fingerprint TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                account TEXT NOT NULL,
                provider_task_id TEXT,
                provider_status TEXT,
                estimate TEXT,
                provider_url TEXT,
                result_path TEXT,
                actual_metadata TEXT,
                project TEXT,
                error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(account, status, created_at)")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "download_attempts" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN download_attempts INTEGER NOT NULL DEFAULT 0")
            if "actual_metadata" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN actual_metadata TEXT")
            if "project" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN project TEXT")
            for name, kind in (("shot", "TEXT"), ("input_metadata", "TEXT"),
                               ("last_polled_at", "REAL"), ("started_at", "REAL"),
                               ("completed_at", "REAL"), ("submitted_at", "REAL"),
                               ("failure_details", "TEXT"), ("generation_params", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project, created_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS job_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                at REAL NOT NULL, kind TEXT NOT NULL, details TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS events_job ON job_events(job_id, seq)")
            db.execute("""CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE, requires_binding INTEGER NOT NULL
            )""")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity_salt', ?)", (uuid.uuid4().hex,))
            # Old Windows paths are normalized in-place, in one transaction.
            for column in ("first_frame", "last_frame", "result_path"):
                db.execute(f"UPDATE jobs SET {column}=replace({column}, ?, '/') WHERE instr({column}, ?) > 0",
                           ("\\", "\\"))
        self.db_path.chmod(0o600)

    def configure_accounts(self, accounts):
        with self._connect() as db:
            for account in accounts:
                db.execute("""INSERT OR IGNORE INTO accounts (id, requires_binding)
                    VALUES (?, EXISTS(SELECT 1 FROM jobs WHERE account=?))""", (account, account))

    def verify_identity(self, account, identity, *, bind_legacy=False):
        with self._connect() as db:
            # Serialize first-time binding across workers; one login cannot serve two queues.
            db.execute("BEGIN IMMEDIATE")
            salt = db.execute("SELECT value FROM metadata WHERE key='identity_salt'").fetchone()[0]
            fingerprint = hmac.new(salt.encode(), identity.encode(), hashlib.sha256).hexdigest()
            row = db.execute("SELECT * FROM accounts WHERE id=?", (account,)).fetchone()
            if row["fingerprint"]:
                if not hmac.compare_digest(row["fingerprint"], fingerprint):
                    raise ServiceError(f"ModelScope identity mismatch for {account}; restore its original login",
                                       409, "account_identity_mismatch")
                return
            if row["requires_binding"] and not bind_legacy:
                raise ServiceError(f"Legacy account {account} needs explicit --bind-accounts before resuming",
                                   409, "account_binding_required")
            try:
                db.execute("UPDATE accounts SET fingerprint=?, requires_binding=0 WHERE id=?",
                           (fingerprint, account))
            except sqlite3.IntegrityError as exc:
                raise ServiceError("This ModelScope login is already bound to another account ID",
                                   409, "duplicate_account_identity") from exc

    def account_for_identity(self, identity):
        """Resolve a verified login without changing any existing job ownership."""
        with self._connect() as db:
            salt = db.execute("SELECT value FROM metadata WHERE key='identity_salt'").fetchone()[0]
            fingerprint = hmac.new(salt.encode(), identity.encode(), hashlib.sha256).hexdigest()
            row = db.execute("SELECT id FROM accounts WHERE fingerprint=?", (fingerprint,)).fetchone()
            return row["id"] if row else None

    def recover(self):
        now = time.time()
        with self._connect() as db:
            interrupted = db.execute("SELECT id,status FROM jobs WHERE status='uploading' OR (status='submitting' AND provider_task_id IS NULL)").fetchall()
            db.execute("UPDATE jobs SET status='queued', updated_at=?, error=NULL WHERE status='uploading'", (now,))
            db.execute("""UPDATE jobs SET status='unknown_submission', updated_at=?,
                error='The service stopped during submission; check ModelScope before retrying.'
                WHERE status='submitting' AND provider_task_id IS NULL""", (now,))
            for job in interrupted:
                self._event(db, job["id"], "service_recovery", {"previous_status": job["status"],
                            "action": "requeue_unsubmitted" if job["status"] == "uploading" else "reconcile_required"})

    def get_by_idempotency(self, key):
        if not key:
            return None
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
        return dict(row) if row else None

    def active_accounts(self):
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT DISTINCT account FROM jobs WHERE status IN ({placeholders})",
                ACTIVE_STATES,
            ).fetchall()
        return {row["account"] for row in rows}

    def active_counts(self, accounts):
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT account, COUNT(*) AS n FROM jobs WHERE status IN ({placeholders}) GROUP BY account",
                ACTIVE_STATES,
            ).fetchall()
        counts = {account: 0 for account in accounts}
        counts.update({row["account"]: row["n"] for row in rows})
        return counts

    def insert(self, row):
        columns = tuple(row)
        placeholders = ",".join("?" for _ in columns)
        with self._connect() as db:
            db.execute(f"INSERT INTO jobs ({','.join(columns)}) VALUES ({placeholders})", tuple(row.values()))
            self._event(db, row["id"], "queued", {"account": row["account"]})

    @staticmethod
    def _event(db, job_id, kind, details):
        db.execute("INSERT INTO job_events(job_id,at,kind,details) VALUES(?,?,?,?)",
                   (job_id, time.time(), kind, json.dumps(details, ensure_ascii=False)))

    def events(self, job_id):
        with self._connect() as db:
            rows = db.execute("SELECT at,kind,details FROM job_events WHERE job_id=? ORDER BY seq DESC LIMIT 100",
                              (job_id,)).fetchall()
        return [{"at": r["at"], "kind": r["kind"], **json.loads(r["details"])} for r in reversed(rows)]

    def get(self, job_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, project=None, limit=100, offset=0):
        with self._connect() as db:
            if project:
                rows = db.execute("SELECT * FROM jobs WHERE project=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                                  (project, limit, offset)).fetchall()
            else:
                rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                                  (limit, offset)).fetchall()
        return [dict(row) for row in rows]

    def queue_position(self, job):
        if job["status"] != "queued":
            return None
        with self._connect() as db:
            return db.execute("""SELECT COUNT(*) FROM jobs WHERE account=? AND
                (status IN ('uploading','submitting','in_progress','download_pending','unknown_submission')
                 OR (status='queued' AND (created_at<? OR (created_at=? AND id<=?))))""",
                (job["account"], job["created_at"], job["created_at"], job["id"])).fetchone()[0]

    def update(self, job_id, **fields):
        if not fields:
            return
        fields["updated_at"] = time.time()
        clause = ", ".join(f"{name}=?" for name in fields)
        with self._connect() as db:
            previous = db.execute("SELECT status,account FROM jobs WHERE id=?", (job_id,)).fetchone()
            db.execute(f"UPDATE jobs SET {clause} WHERE id=?", (*fields.values(), job_id))
            if previous and (fields.get("status", previous["status"]) != previous["status"] or
                             fields.get("account", previous["account"]) != previous["account"]):
                self._event(db, job_id, fields.get("status", previous["status"]),
                            {"account": fields.get("account", previous["account"]),
                             "previous_account": previous["account"],
                             "provider_task_id": fields.get("provider_task_id")})

    def record_event(self, job_id, kind, **details):
        with self._connect() as db:
            self._event(db, job_id, kind, details)

    def account_jobs(self, account):
        with self._connect() as db:
            placeholders = ','.join('?' for _ in ACTIVE_STATES)
            rows = db.execute(f"SELECT * FROM jobs WHERE account=? AND status IN ({placeholders}) ORDER BY created_at,id",
                              (account, *ACTIVE_STATES)).fetchall()
        return [dict(row) for row in rows]

    def next_for_account(self, account):
        with self._connect() as db:
            blocked = db.execute(
                "SELECT 1 FROM jobs WHERE account=? AND status IN ('submitting', 'unknown_submission') LIMIT 1",
                (account,),
            ).fetchone()
            if blocked:
                return None
            row = db.execute("""SELECT * FROM jobs WHERE account=?
                AND status IN ('in_progress', 'download_pending') AND provider_task_id IS NOT NULL
                ORDER BY created_at LIMIT 1""", (account,)).fetchone()
            if row is None:
                row = db.execute("SELECT * FROM jobs WHERE account=? AND status='queued' ORDER BY created_at LIMIT 1",
                                 (account,)).fetchone()
        return dict(row) if row else None


class WanService:
    def __init__(self, *, data_dir=None, accounts=None, media_roots=None,
                 poll_seconds=15):
        account_urls = accounts if accounts is not None else parse_accounts()
        configured_accounts = set(account_urls) if isinstance(account_urls, dict) else {
            ms_provider.account_label(index) for index in range(len(account_urls))
        }
        self.account_urls = dict(account_urls) if isinstance(account_urls, dict) else {
            ms_provider.account_label(index): url for index, url in enumerate(account_urls)}
        if self.account_urls:
            validate_accounts(self.account_urls)
        configured_roots = (tuple(Path(p).expanduser().resolve() for p in media_roots)
                            if media_roots is not None else tuple(parse_media_roots()))
        if not configured_roots:
            raise ServiceError("WAN_MEDIA_ROOTS must contain at least one directory")
        self.store = JobStore(data_dir or default_data_dir())
        registry_path = self.store.data_dir / "browser-accounts.json"
        try:
            saved_accounts = json.loads(registry_path.read_text(encoding="utf-8"))
            if not isinstance(saved_accounts, dict):
                raise ServiceError("Saved browser account registry is invalid")
            configured_ports = {urlsplit(url).port for url in self.account_urls.values()
                                if urlsplit(url).port is not None}
            for account, url in (validate_accounts(saved_accounts) if saved_accounts else {}).items():
                if account in self.account_urls and (self.store.data_dir / "browser-profiles" / account).is_dir():
                    # The registry contains the last dynamically allocated port.
                    # A configured managed account will receive a new port at start.
                    continue
                if account in self.account_urls and self.account_urls[account] != url:
                    raise ServiceError(f"Saved browser account conflicts with configured account {account}")
                # Managed browser ports are ephemeral. If an old registry points
                # at a port now reserved by a manually configured account, assign
                # a fresh local port before validating the merged account map.
                if account not in self.account_urls and urlsplit(url).port in configured_ports:
                    url = f"http://127.0.0.1:{_free_cdp_port(excluded=configured_ports)}"
                self.account_urls.setdefault(account, url)
                if account not in configured_accounts:
                    configured_ports.add(urlsplit(url).port)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, TypeError) as exc:
            raise ServiceError("Saved browser account registry is invalid") from exc
        if self.account_urls:
            validate_accounts(self.account_urls)
        # A local account profile marks a configured endpoint as service-managed.
        # This lets launchd restart its Chrome and keeps manual CDP endpoints
        # externally managed when no private profile exists.
        managed_accounts = {
            account for account in configured_accounts
            if (self.store.data_dir / "browser-profiles" / account).is_dir()
        }
        self._startup_accounts = configured_accounts - managed_accounts
        lock_path = self.store.data_dir / ".service.lock"
        self._service_lock = lock_path.open("a+b")
        os.chmod(lock_path, 0o600)
        try:
            if os.name == "nt":
                if lock_path.stat().st_size == 0:
                    self._service_lock.write(b"\0")
                    self._service_lock.flush()
                self._service_lock.seek(0)
                msvcrt.locking(self._service_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._service_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._service_lock.close()
            raise ServiceError("another WAN service is already running", 409,
                               "service_already_running") from exc
        self.accounts = list(self.account_urls)
        self.account_labels = {account: account for account in self.accounts}
        self._profile_lock = threading.Lock()
        self._accounts_lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._browser_processes = {}
        self._browser_modes = {}
        self._pending_accounts = set()
        self._account_unavailable_since = {}
        self._account_operations = {account: threading.Lock() for account in self.accounts}
        self.account_profiles = {account: {"id": account, "label": account, "balance": None,
                                          "status": "connecting", "checked_at": None,
                                          "unit": "魔粒", "balance_error": None,
                                          "session_error": None}
                                 for account in self.accounts}
        self.store.configure_accounts(self.accounts)
        self.media_roots = configured_roots
        self.poll_seconds = max(2, int(poll_seconds))
        self.stop_event = threading.Event()
        self.workers = []
        from .login import LoginManager
        self.logins = LoginManager(self)
        self.store.recover()
        missing_accounts = self.store.active_accounts() - set(self.accounts)
        if missing_accounts:
            self._service_lock.close()
            labels = ", ".join(sorted(missing_accounts))
            raise ServiceError(
                f"Active jobs require the previously configured ModelScope sessions: {labels}",
                409,
                "missing_job_account",
            )

    @staticmethod
    def _generation_params(body, model=None):
        result = {}
        for name, default in (("accelerate_lora", model == "minimax-h3"), ("add_watermark", False)):
            value = body.get(name, default)
            if not isinstance(value, bool):
                raise ServiceError(f"{name} must be a boolean")
            result[name] = value
        for name in ("checkpoint_version_id", "vae_version_id", "num_inference_steps"):
            value = body.get(name)
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool) or value < 1 or (name == "num_inference_steps" and value > 50):
                    raise ServiceError(f"{name} must be a positive integer" + (" from 1 to 50" if name == "num_inference_steps" else ""))
                result[name] = value
        value = body.get("guidance_scale")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1.5 <= value <= 20:
                raise ServiceError("guidance_scale must be from 1.5 to 20")
            result["guidance_scale"] = value
        if result["accelerate_lora"] and "num_inference_steps" in result and result["num_inference_steps"] != 4:
            raise ServiceError("accelerate_lora uses 4 steps; disable it to set num_inference_steps")
        loras = body.get("loras", [])
        if not isinstance(loras, list) or len(loras) > 6:
            raise ServiceError("loras must be a list of at most 6 entries")
        clean, seen = [], set()
        for entry in loras:
            if not isinstance(entry, dict) or set(entry) - {"model_version_id", "scale", "trigger_words"}:
                raise ServiceError("LoRA entries accept model_version_id, scale and trigger_words")
            version, scale, words = entry.get("model_version_id"), entry.get("scale", 1), entry.get("trigger_words", [])
            if isinstance(version, bool) or not isinstance(version, int) or version < 1 or version in seen:
                raise ServiceError("LoRA model_version_id must be a unique positive integer")
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not -10 <= scale <= 10:
                raise ServiceError("LoRA scale must be from -10 to 10")
            if not isinstance(words, list) or len(words) > 32 or any(not isinstance(w, str) or len(w) > 1000 for w in words):
                raise ServiceError("LoRA trigger_words must contain at most 32 strings of at most 1000 characters")
            seen.add(version)
            clean.append({"model_version_id": version, "scale": scale, "trigger_words": words})
        result["loras"] = clean
        return result

    @staticmethod
    def _normalize_model(model):
        if isinstance(model, str) and "/" in model:
            model = model.rsplit("/", 1)[-1]
        if not isinstance(model, str) or model not in MODEL_ALIASES:
            choices = ", ".join(MODEL_IDS)
            raise ServiceError(f"model must be one of: {choices}")
        return model if model in MODEL_IDS else next(k for k, v in MODEL_IDS.items() if v == model)

    def account_snapshot(self, account=None):
        with self._profile_lock:
            records = [dict(value) for value in self.account_profiles.values()]
        now = time.time()
        for value in records:
            value["stale"] = value["checked_at"] is None or now - value["checked_at"] > 120
            if value["stale"]:
                value["balance"] = None
        if account is None:
            return records
        result = next((value for value in records if value["id"] == account), None)
        if result is None:
            raise ServiceError("Account not found.", 404, "not_found_error")
        return result

    def _refresh_account(self, ms, account):
        # Never publish a name/balance before verifying the bound browser identity.
        self.store.verify_identity(account, ms.identity())
        label = ms.profile()
        balance, balance_error = None, None
        try:
            raw = ms.balance()
            raw = raw.get("available_balance") if isinstance(raw, dict) else raw
            if isinstance(raw, bool) or raw is None:
                raise ValueError("No available balance")
            balance = float(raw)
            if not math.isfinite(balance) or balance < 0:
                raise ValueError("Invalid available balance")
        except Exception as exc:
            balance = None
            balance_error = str(exc) if isinstance(exc, RuntimeError) and "ModelScope" in str(exc) \
                else "魔粒余额暂不可用；稍后自动重试"
        self.store.verify_identity(account, ms.identity())
        with self._profile_lock:
            self._account_unavailable_since.pop(account, None)
            self.account_profiles[account] = {
                "id": account, "label": label, "balance": balance, "unit": "魔粒",
                "status": "connected", "checked_at": time.time(), "balance_error": balance_error,
                "session_error": None}

    def validate_frames(self, body):
        if not body.get("first_frame"):
            raise ServiceError("first_frame is required")
        frames = {}
        for field in ("first_frame", "last_frame"):
            if body.get(field) is not None:
                _, width, height = _validate_frame(body[field], self.media_roots)
                frames[field] = {"width": width, "height": height}
        return {"valid": True, "frames": frames}

    def submit(self, body, idempotency_header=None):
        # Selection and durable insertion are one scheduling operation.
        with self._submit_lock:
            return self._submit(body, idempotency_header)

    def _submit(self, body, idempotency_header=None, *, quote_only=False):
        if not isinstance(body, dict):
            raise ServiceError("request body must be a JSON object")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ServiceError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ServiceError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")
        project = body.get("project")
        if project is not None:
            if not isinstance(project, str) or not project.strip() or len(project) > MAX_PROJECT_CHARS:
                raise ServiceError(f"project must be a non-empty string of at most {MAX_PROJECT_CHARS} characters")
            project = project.strip()
        shot = body.get("shot")
        if shot is not None:
            if not isinstance(shot, str) or not shot.strip() or len(shot) > MAX_PROJECT_CHARS:
                raise ServiceError("shot must be a non-empty string of at most 160 characters")
            shot = shot.strip()
        requested_account = body.get("account")
        if requested_account is not None and (not isinstance(requested_account, str) or
                                               requested_account not in self.accounts):
            raise ServiceError("account must be one of the configured account IDs")

        routing = body.get("routing")
        if routing not in (None, "preferred", "balanced"):
            raise ServiceError("routing must be preferred or balanced")
        preferred_account = body.get("preferred_account")
        if preferred_account is not None and (not isinstance(preferred_account, str) or
                                               preferred_account not in self.accounts):
            raise ServiceError("preferred_account must be one of the configured account IDs")

        model = self._normalize_model(body.get("model", DEFAULT_MODEL))
        mode = body.get("mode", "i2v")
        if mode not in ("i2v", "t2v"):
            raise ServiceError("mode must be i2v or t2v")
        if mode == "t2v" and model not in TEXT_MODELS:
            raise ServiceError("this model does not support text-to-video")
        if mode == "i2v" and model not in IMAGE_MODELS:
            raise ServiceError("this model requires text-to-video mode")
        first_value = body.get("first_frame", body.get("image"))
        last_value = body.get("last_frame")
        first_path = last_path = None
        if mode == "t2v":
            if first_value is not None or last_value is not None:
                raise ServiceError("text-to-video does not accept input frames")
            width, height = _parse_size(body.get("size")) or (1280, 720)
        else:
            if first_value is None:
                raise ServiceError("first_frame is required; use an absolute local image path")
            first_path, source_w, source_h = _validate_frame(first_value, self.media_roots)
            if last_value is not None:
                last_path, last_w, last_h = _validate_frame(last_value, self.media_roots)
                if abs(last_w / last_h - source_w / source_h) > 0.015:
                    raise ServiceError("first_frame and last_frame must have matching aspect ratios")
            width, height = _parse_size(body.get("size")) or _infer_dimensions(first_path)
            if abs(width / height - source_w / source_h) > 0.015:
                raise ServiceError("size must preserve the first frame's aspect ratio")

        duration_value = body.get("seconds", body.get("duration", 5))
        if isinstance(duration_value, bool):
            raise ServiceError("seconds must be a whole number from 1 to 10")
        try:
            duration_float = float(duration_value)
        except (TypeError, ValueError) as exc:
            raise ServiceError("seconds must be a whole number from 1 to 10") from exc
        if not duration_float.is_integer() or not 1 <= duration_float <= MAX_DURATION_SECONDS:
            raise ServiceError("seconds must be a whole number from 1 to 10")
        seconds = int(duration_float)
        if last_path is not None and model not in LAST_FRAME_MODELS:
            raise ServiceError("last_frame is supported only by wan2.1-flf2v-14b, minimax-h3 and ltx-2.3")
        style = body.get("style")
        negative_prompt = body.get("negative_prompt")
        for name, value in (("style", style), ("negative_prompt", negative_prompt)):
            if value is not None and not isinstance(value, str):
                raise ServiceError(f"{name} must be a string when provided")
            if isinstance(value, str) and len(value) > MAX_STYLE_CHARS:
                raise ServiceError(f"{name} must be at most {MAX_STYLE_CHARS} characters")
        seed = body.get("seed", -1)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1 or seed > 2_147_483_647:
            raise ServiceError("seed must be -1 or a non-negative 32-bit integer")
        generation_params = self._generation_params(body, model)
        generation_params["mode"] = mode
        if quote_only:
            return {"model": model, "prompt": prompt, "seconds": seconds, "width": width, "height": height,
                    "style": style, "negative_prompt": negative_prompt, "seed": seed,
                    "generation_params": generation_params, "account": requested_account}
        supplied_metadata = body.get("input_metadata") or {}
        if not isinstance(supplied_metadata, dict):
            raise ServiceError("input_metadata must be an object")
        idempotency_key = idempotency_header or body.get("idempotency_key")
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not idempotency_key.strip() or len(idempotency_key) > 200:
                raise ServiceError("idempotency_key must be a non-empty string of at most 200 characters")
            idempotency_key = idempotency_key.strip()

        fingerprint = _fingerprint(first_path, last_path, prompt, model, seconds, width, height,
                                   style, negative_prompt, seed, project, shot, requested_account, routing, preferred_account, generation_params)
        previous = self.store.get_by_idempotency(idempotency_key)
        if previous:
            if previous["input_fingerprint"] != fingerprint:
                # A retry of a pre-VideoGen request must still return its old job
                # even though new requests now have explicit generation defaults.
                advanced = {"mode", "accelerate_lora", "num_inference_steps", "guidance_scale", "loras",
                            "checkpoint_version_id", "vae_version_id", "add_watermark"}
                legacy = _fingerprint(first_path, last_path, prompt, model, seconds, width, height,
                                      style, negative_prompt, seed, project, shot, requested_account, routing, preferred_account)
                if previous.get("generation_params") or advanced.intersection(body) or previous["input_fingerprint"] != legacy:
                    raise ServiceError("idempotency key was already used for different input", 409, "idempotency_conflict")
            return previous, False

        job_id = "wan_" + uuid.uuid4().hex
        job_dir = self.store.jobs_dir / job_id
        input_dir = job_dir / "inputs"
        job_dir.mkdir(mode=0o700)
        input_dir.mkdir(mode=0o700)
        first_copy = None
        if first_path:
            first_copy = input_dir / ("first" + (first_path.suffix.lower() or ".img"))
            shutil.copyfile(first_path, first_copy)
            first_copy.chmod(0o600)
        last_copy = None
        if last_path is not None:
            last_copy = input_dir / ("last" + (last_path.suffix.lower() or ".img"))
            shutil.copyfile(last_path, last_copy)
            last_copy.chmod(0o600)
        stored_fingerprint = _fingerprint(first_copy, last_copy, prompt, model, seconds, width, height,
                                          style, negative_prompt, seed, project, shot, requested_account, routing, preferred_account, generation_params)

        eligible = [a for a in self.accounts if a not in self._pending_accounts
                    and not self.logins.is_active(a)]
        if requested_account and requested_account not in eligible or not requested_account and not eligible:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ServiceError("账号正在登录，请完成登录后再提交任务。", 409, "account_login_pending")
        counts = self.store.active_counts(eligible)
        if requested_account:
            account = requested_account
        elif routing is None:
            # Preserve the legacy HTTP API's queue-only scheduling.
            account = min(eligible, key=lambda candidate: counts.get(candidate, 0))
        else:
            profiles = {p["id"]: p for p in self.account_snapshot()}
            available = [a for a in eligible if profiles[a]["status"] == "connected"
                         and not profiles[a]["stale"]
                         and isinstance(profiles[a]["balance"], (int, float))
                         and not isinstance(profiles[a]["balance"], bool)
                         and math.isfinite(profiles[a]["balance"]) and profiles[a]["balance"] > 0]
            if not available:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise ServiceError("没有余额有效且已登录的账号，请刷新余额或重新登录。", 409,
                                   "no_available_account")
            if routing == "preferred":
                account = preferred_account or available[0]
                if account not in available:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise ServiceError("首选账号当前不可用，请更换首选账号或使用均衡模式。", 409,
                                       "preferred_account_unavailable")
            else:
                # Equivalent to a max-heap top; do not invent deductions before
                # ModelScope reports the actual available wallet balance.
                account = min(available, key=lambda a: (-profiles[a]["balance"], counts[a], a))
        input_metadata = {}
        for role, path, w, h in (("first_frame", first_path, source_w if first_path else None, source_h if first_path else None),
                                 ("last_frame", last_path, last_w if last_path else None,
                                  last_h if last_path else None)):
            if path:
                supplied = supplied_metadata.get(role) or {}
                name = supplied.get("name") if isinstance(supplied, dict) else None
                if not isinstance(name, str) or not name.strip() or len(name) > 255 or Path(name).name != name:
                    name = path.name
                input_metadata[role] = {"name": name, "width": w, "height": h}
        now = time.time()
        row = {
            "id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "prompt": prompt,
            "first_frame": self.store.relative_data_path(first_copy) if first_copy else "",
            "last_frame": self.store.relative_data_path(last_copy) if last_copy else None,
            "model": model,
            "seconds": seconds,
            "width": width,
            "height": height,
            "style": style,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "input_fingerprint": stored_fingerprint,
            "idempotency_key": idempotency_key,
            "account": account,
            "project": project,
            "shot": shot,
            "input_metadata": json.dumps(input_metadata, ensure_ascii=False),
            "generation_params": json.dumps(generation_params, ensure_ascii=False),
        }
        try:
            self.store.insert(row)
        except sqlite3.IntegrityError:
            shutil.rmtree(job_dir, ignore_errors=True)
            previous = self.store.get_by_idempotency(idempotency_key)
            if previous and previous["input_fingerprint"] == stored_fingerprint:
                return previous, False
            raise ServiceError("could not create job because the idempotency key is already in use", 409,
                               "idempotency_conflict")
        return self.store.get(job_id), True

    def public_job(self, job, base_path="/v1/videos"):
        internal_status = job["status"]
        status = PUBLIC_STATUS[internal_status]
        # ModelScope's status API does not promise a measured completion percent.
        progress = 100 if internal_status == "completed" else None
        phase = {
            "queued": "Waiting for an account worker",
            "uploading": "Uploading input frames",
            "submitting": "Submitting to ModelScope",
            "in_progress": "Generating on ModelScope",
            "download_pending": "Fetching generated video",
            "unknown_submission": "Submission needs reconciliation",
            "completed": "Video ready",
            "failed": "Generation failed",
            "paused": "Local queue paused",
            "cancelled": "Local waiting task cancelled",
        }[internal_status]
        estimate = json.loads(job["estimate"]) if job.get("estimate") else None
        now = time.time()
        last_poll = job.get("last_polled_at")
        terminal = internal_status in ("completed", "failed", "cancelled")
        elapsed_end = job.get("completed_at") or (job["updated_at"] if terminal else now)
        result = {
            "id": job["id"],
            "object": "video",
            "model": job["model"],
            "status": status,
            "progress": progress,
            "phase": phase,
            "progress_kind": "complete" if internal_status == "completed" else "indeterminate",
            "progress_message": (job.get("error") or phase) + (
                ". ModelScope does not provide a reliable percentage or remaining time."
                if internal_status == "in_progress" else ""),
            "estimated_remaining_seconds": None,
            "elapsed_seconds": max(0, int(elapsed_end - job["created_at"])),
            "last_polled_at": int(last_poll) if last_poll else None,
            "poll_age_seconds": max(0, int(now - last_poll)) if last_poll else None,
            "status_stale": bool(last_poll and not terminal and now - last_poll > max(60, self.poll_seconds * 3)),
            "poll_after_seconds": None if terminal else self.poll_seconds,
            "queue_position": self.store.queue_position(job),
            "created_at": int(job["created_at"]),
            "updated_at": int(job["updated_at"]),
            "seconds": job["seconds"],
            "size": f"{job['width']}x{job['height']}",
            "requested": {"seconds": job["seconds"], "size": f"{job['width']}x{job['height']}",
                          "width": job["width"], "height": job["height"], "fps": None},
            "project": job.get("project"),
            "shot": job.get("shot"),
            "account": self.account_labels.get(job["account"], "unconfigured-account"),
            "account_id": job["account"],
            "provider_task_id": job.get("provider_task_id"),
            "estimate": estimate,
            "error": job.get("error"),
            "internal_status": internal_status,
            "provider_status": job.get("provider_status"),
        }
        result["estimate_unit"] = "魔粒"
        profile = next((p for p in self.account_snapshot() if p["id"] == job["account"]), None)
        connection = "unconfigured" if profile is None else profile["status"]
        unavailable = profile is None or connection != "connected" or profile["stale"]
        peers = self.store.account_jobs(job["account"])
        uncertain = next((p for p in peers if p["status"] in ("submitting", "unknown_submission")), None)
        next_job = self.store.next_for_account(job["account"])
        blocker = uncertain or (next_job if next_job and next_job["id"] != job["id"] else None)
        reason = None
        if not terminal:
            if internal_status == "paused":
                reason = "此任务已暂停，恢复后才会提交。"
            elif internal_status == "unknown_submission":
                reason = "提交结果不明，需要核对原账号的服务端任务；不会自动重发。"
            elif unavailable:
                reason = "账号连接不可用或状态过期，等待账号恢复。"
            elif internal_status == "queued" and blocker:
                reason = f"等待同账号任务 {blocker['id']}（{blocker['status']}）。"
            elif result["status_stale"]:
                reason = "任务状态查询已过期，等待下一次成功查询。"
        waiting = [p["id"] for p in peers if p["status"] == "queued" and p["id"] != job["id"]]
        result["connection"] = {"status": connection, "stale": profile["stale"] if profile else True,
                                "error": profile.get("session_error") if profile else "账号未配置",
                                "unavailable_since": self._account_unavailable_since.get(job["account"])}
        holds_queue = not terminal and ((next_job and next_job["id"] == job["id"]) or
                                        internal_status == "unknown_submission")
        result["blockage"] = {"reason": reason, "blocked_by_job_id": blocker["id"] if blocker and internal_status == "queued" else None,
                              "waiting_job_ids": waiting if holds_queue else []}
        result["last_confirmed_provider_status"] = job.get("provider_status")
        result["execution_phase"] = phase
        if reason:
            result["phase"] = reason
            result["progress_message"] = reason + (f" 服务端最后状态：{job['provider_status']}。" if job.get("provider_status") else "")
        queued_end = job.get("started_at") or (elapsed_end if terminal else now)
        submitted = job.get("submitted_at")
        generation_end = job.get("completed_at") or (job["updated_at"] if terminal else now)
        result["timing"] = {"queue_seconds": max(0, int(queued_end - job["created_at"])),
                            "generation_seconds": max(0, int(generation_end - submitted)) if submitted else None,
                            "since_last_successful_poll_seconds": result["poll_age_seconds"],
                            "connection_unavailable_seconds": max(0, int(now - result["connection"]["unavailable_since"]))
                            if result["connection"]["unavailable_since"] else None}
        result["failure"] = json.loads(job["failure_details"]) if job.get("failure_details") else None
        result["recovery"] = {"safe_to_resubmit": internal_status == "failed" and not job.get("provider_task_id"),
                              "action": "reconcile" if internal_status == "unknown_submission" else
                              "restore_account" if not terminal and unavailable else
                              "resume" if internal_status == "paused" else
                              "inspect_provider_task_before_retry" if internal_status == "failed" and job.get("provider_task_id") else None,
                              "actual_charge": None, "charge_verified": False}
        result["quality_acceptance"] = "not_evaluated"
        result["estimate_kind"] = "estimated_cost_not_duration_or_final_charge"
        result["actual"] = None
        result["spec_warnings"] = []
        if job.get("actual_metadata"):
            actual = json.loads(job["actual_metadata"])
            result["actual"] = {key: actual.get(key) for key in
                                 ("duration", "width", "height", "fps", "frame_count")}
            if actual.get("duration") and abs(actual["duration"] - job["seconds"]) > 0.1:
                result["spec_warnings"].append(
                    f"Requested {job['seconds']}s; actual {actual['duration']:.3f}s. Adjust timing in the editor.")
            if actual.get("width") and actual.get("height") and (actual["width"], actual["height"]) != (job["width"], job["height"]):
                result["spec_warnings"].append(
                    f"Requested {job['width']}x{job['height']}; actual {actual['width']}x{actual['height']}. Review cropping or padding.")
        result["prompt"] = job["prompt"]
        result["generation_params"] = json.loads(job.get("generation_params") or "{}")
        result["style"] = job.get("style")
        result["negative_prompt"] = job.get("negative_prompt")
        metadata = json.loads(job.get("input_metadata") or "{}")
        result["input_frames"] = [
            {"role": field, **metadata.get(field, {}),
             "preview_url": f"{base_path}/{job['id']}/frames/{field}"}
            for field in ("first_frame", "last_frame") if job.get(field)]
        if job.get("result_path"):
            result["output_path"] = job["result_path"]
            result["content_url"] = f"{base_path}/{job['id']}/content"
        return result

    def job_details(self, job):
        """Backfill old completed outputs when explicitly inspected, without regenerating."""
        if job["status"] == "completed" and job.get("result_path") and not job.get("actual_metadata"):
            from .media import InvalidVideoError, validate_video_file
            try:
                actual = validate_video_file(self.store.resolve_data_path(job["result_path"]))
                # Preserve the original completion/update time when migrating metadata.
                with self.store._connect() as db:
                    db.execute("UPDATE jobs SET actual_metadata=? WHERE id=?",
                               (json.dumps(actual), job["id"]))
                job = self.store.get(job["id"])
            except (InvalidVideoError, ServiceError):
                pass
        result = self.public_job(job)
        result["events"] = self.store.events(job["id"])
        return result

    def control_job(self, job_id, body):
        action = body.get("action")
        if action not in ("pause", "resume", "cancel", "reassign", "resolve"):
            raise ServiceError("action must be pause, resume, cancel, reassign or resolve")
        with self._submit_lock:
            job = self.store.get(job_id)
            if job is None:
                raise ServiceError("video job not found", 404, "not_found_error")
            operation = self._account_operations[job["account"]]
            if not operation.acquire(blocking=False):
                raise ServiceError("账号正在操作任务，请稍后重试。", 409, "account_busy")
            try:
                job = self.store.get(job_id)
                if action == "resolve":
                    return self.resolve_unknown_submission(job_id, body)
                if job["status"] not in ("queued", "paused") or job.get("provider_task_id"):
                    raise ServiceError("仅尚未提交的 queued/paused 任务可执行此操作。", 409, "invalid_job_state")
                if action == "reassign":
                    target = body.get("account")
                    if not isinstance(target, str) or target not in self.accounts:
                        raise ServiceError("account must be a configured account ID")
                    p = self.account_snapshot(target)
                    if target in self._pending_accounts or self.logins.is_active(target) or p["status"] != "connected" or p["stale"]:
                        raise ServiceError("目标账号当前不可用。", 409, "account_unavailable")
                    self.store.update(job_id, account=target, error=None)
                else:
                    self.store.update(job_id, status={"pause": "paused", "resume": "queued", "cancel": "cancelled"}[action],
                                      completed_at=time.time() if action == "cancel" else None, error=None)
                return self.store.get(job_id)
            finally:
                operation.release()

    def diagnostics(self):
        accounts = self.account_snapshot()
        with self.store._connect() as db:
            rows = db.execute("SELECT id,account,status FROM jobs WHERE status IN ('queued','paused','unknown_submission','in_progress','download_pending','submitting','uploading')").fetchall()
        unhealthy = {a["id"] for a in accounts if a["status"] != "connected" or a["stale"]}
        return {"service_version": __version__, "protocol_version": PROTOCOL_VERSION,
                "workers": [{"account": a, "alive": any(w.name == f"wan-worker-{a}" and w.is_alive() for w in self.workers)} for a in self.accounts],
                "accounts": accounts, "media_roots": [{"path": str(p), "exists": p.is_dir()} for p in self.media_roots],
                "affected_jobs": [dict(r) for r in rows if r["account"] in unhealthy or r["status"] == "unknown_submission"]}

    def models(self):
        with self.store._connect() as db:
            rows = db.execute("SELECT model,MAX(completed_at) AS last_success FROM jobs WHERE status='completed' GROUP BY model").fetchall()
        successes = {r["model"]: r["last_success"] for r in rows}
        return {"object": "list", "data": [
            {"id": alias, "object": "model", "created": 0, "owned_by": "modelscope", "kind": "video",
             "first_frame_required": alias not in TEXT_MODELS, "image_to_video_supported": alias in IMAGE_MODELS,
             "text_to_video_supported": alias in TEXT_MODELS, "last_frame_supported": alias in LAST_FRAME_MODELS,
             "acceleration_default": alias == "minimax-h3", "acceleration_steps": 4,
             "generation_parameters": ["accelerate_lora", "num_inference_steps", "guidance_scale", "loras",
                                       "checkpoint_version_id", "vae_version_id", "add_watermark", "seed", "negative_prompt"],
             "seconds": {"min": 1, "max": MAX_DURATION_SECONDS},
             "dimensions": {"min": 256, "max": 1536, "multiple_of": 16, "preserve_aspect_ratio": True},
             "limits_source": "adapter_validation_not_live_provider_guarantee",
             "provider_verified_at": None, "last_local_download_success_at": successes.get(alias),
             "quality_acceptance": "not_evaluated"} for alias in MODEL_IDS]}

    def provider_query(self, body, *, catalog=False):
        """Short read/quote operations, serialized with the original account worker."""
        account = body.get("account") or next((a for a in self.accounts
                                              if self.account_snapshot(a)["status"] == "connected"), None)
        if account not in self.accounts or self.logins.is_active(account):
            raise ServiceError("No connected ModelScope account is available", 409)
        lock = self._account_operations[account]
        if not lock.acquire(timeout=1):
            raise ServiceError("Account is busy uploading/submitting; retry shortly", 409)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                ms = ms_provider.connect(playwright, self.account_urls[account])
                self.store.verify_identity(account, ms.identity())
                if catalog:
                    model = self._normalize_model(body.get("model", "minimax-h3"))
                    kind = body.get("model_type", "LoRA")
                    query = body.get("query", "")
                    page, count = body.get("page", 1), body.get("page_size", 16)
                    if kind not in ("Checkpoint", "LoRA", "VAE") or not isinstance(query, str) or len(query) > 200:
                        raise ServiceError("Invalid model_type or query")
                    if any(isinstance(v, bool) or not isinstance(v, int) for v in (page, count)) or not 1 <= page <= 10000 or not 1 <= count <= 50:
                        raise ServiceError("page must be positive; page_size must be from 1 to 50")
                    return ms.catalog(MODEL_IDS[model], kind, query, page, count)
                spec = self._submit(body, quote_only=True)
                request = ms.request(MODEL_IDS[spec["model"]], None, None, spec["prompt"], spec["seconds"],
                                     width=spec["width"], height=spec["height"], style=spec["style"],
                                     negative_prompt=spec["negative_prompt"], seed=spec["seed"], **spec["generation_params"])
                return {"estimate": ms.estimate(request), "estimate_unit": "魔粒", "account_id": account,
                        "model": spec["model"], "generation_params": spec["generation_params"],
                        "estimate_kind": "estimated_cost_not_duration_or_final_charge", "submitted": False}
        finally:
            lock.release()

    def resolve_unknown_submission(self, job_id, body):
        job = self.store.get(job_id)
        if job is None:
            raise ServiceError("video job not found", 404, "not_found_error")
        if job["status"] != "unknown_submission":
            raise ServiceError("job is not waiting for submission reconciliation", 409, "invalid_job_state")
        provider_task_id = body.get("provider_task_id")
        if provider_task_id is not None and (not isinstance(provider_task_id, str) or
                not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", provider_task_id)):
            raise ServiceError("provider_task_id must contain only letters, numbers, underscores or hyphens")
        if bool(provider_task_id) == (body.get("confirm_no_task") is True):
            raise ServiceError("Choose exactly one: provider_task_id or confirm_no_task=true")
        if isinstance(provider_task_id, str) and provider_task_id.strip():
            self.store.update(job_id, status="in_progress", provider_task_id=provider_task_id.strip(),
                              provider_status=None, provider_url=None, error=None, submitted_at=time.time(), failure_details=None)
        elif body.get("confirm_no_task") is True:
            self.store.update(job_id, status="queued", provider_task_id=None, provider_status=None,
                              provider_url=None, error=None, failure_details=None, submitted_at=None)
        else:
            raise ServiceError("provide provider_task_id, or set confirm_no_task=true after checking ModelScope")
        self.store.record_event(job_id, "reconciled", resolution="linked_task" if provider_task_id else "confirmed_no_task")
        return self.store.get(job_id)

    def start(self):
        for account in self.accounts:
            if account not in self._startup_accounts:
                self._ensure_managed_browser(account)
            self._start_account_worker(account)

    def _start_account_worker(self, account):
        with self._accounts_lock:
            if any(worker.name == f"wan-worker-{self.account_labels[account]}" and worker.is_alive()
                   for worker in self.workers):
                return
            worker = threading.Thread(target=self._worker, args=(account,), daemon=True,
                                      name=f"wan-worker-{self.account_labels[account]}")
            worker.start()
            self.workers.append(worker)

    def _save_browser_accounts(self):
        registry_path = self.store.data_dir / "browser-accounts.json"
        saved = {key: value for key, value in self.account_urls.items()
                 if key not in self._startup_accounts and key not in self._pending_accounts}
        temp = registry_path.with_suffix(".json.partial")
        temp.write_text(json.dumps(saved, sort_keys=True), encoding="utf-8")
        os.chmod(temp, 0o600)
        temp.replace(registry_path)

    def _ensure_managed_browser(self, account, *, headless=True):
        with self._accounts_lock:
            if self.stop_event.is_set():
                raise ServiceError("WAN 服务正在停止。", 503)
            process = self._browser_processes.get(account)
            if process is not None and process.poll() is None:
                if self._browser_modes.get(account) == headless:
                    return
                self._stop_browser(account)
            executable = _browser_executable()
            if executable is None:
                raise ServiceError("未找到 Chrome 或 Chromium。请先安装浏览器后重试。", 503,
                                   "browser_not_found")
            profile = self.store.data_dir / "browser-profiles" / account
            if not profile.is_dir():
                raise ServiceError("账号的浏览器目录不存在，请恢复原目录。", 409)
            # A managed endpoint is local and gets a new free port on each restart.
            excluded_ports = {
                urlsplit(url).port for name, url in self.account_urls.items()
                if name != account and urlsplit(url).port is not None
            }
            port = _free_cdp_port(excluded=excluded_ports)
            command = [str(executable), *(["--headless=new"] if headless else []), "--remote-debugging-address=127.0.0.1",
                       f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                       "--no-first-run", "--no-default-browser-check", "--window-size=1100,760",
                       "https://modelscope.cn"]
            try:
                self._browser_processes[account] = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=(os.name != "nt"))
                self._browser_modes[account] = headless
            except OSError as exc:
                raise ServiceError("无法启动登录浏览器，请重试。", 503, "browser_launch_failed") from exc
            self.account_urls[account] = f"http://127.0.0.1:{port}"
            self._save_browser_accounts()

    def launch_browser_account(self, *, pending=False, headless=True):
        executable = _browser_executable()
        if executable is None:
            raise ServiceError("未找到 Chrome 或 Chromium。请先安装浏览器后重试。", 503,
                               "browser_not_found")
        with self._accounts_lock:
            index = 1
            while (f"account-{index}" in self.account_urls or
                   (self.store.data_dir / "browser-profiles" / f"account-{index}").exists()):
                index += 1
            account = f"account-{index}"
            profile_dir = self.store.data_dir / "browser-profiles" / account
            profile_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            profile_dir.chmod(0o700)
            self._pending_accounts.add(account)
            self.account_urls[account] = "http://127.0.0.1:0"
            self.accounts.append(account)
            self.account_labels[account] = account
            self._account_operations[account] = threading.Lock()
            self.store.configure_accounts([account])
            with self._profile_lock:
                self.account_profiles[account] = {
                    "id": account, "label": account, "balance": None, "unit": "魔粒",
                    "status": "connecting", "checked_at": None, "balance_error": None,
                    "session_error": "请在登录页面中完成 ModelScope 登录。"}
            try:
                self._ensure_managed_browser(account, headless=headless)
                if not pending:
                    self.activate_browser_account(account)
            except Exception:
                self.discard_pending_account(account)
                raise
        return self.account_snapshot(account)

    def activate_browser_account(self, account):
        with self._accounts_lock:
            self._pending_accounts.discard(account)
            self._save_browser_accounts()
            self._start_account_worker(account)

    def discard_pending_account(self, account):
        with self._accounts_lock:
            if account not in self._pending_accounts:
                return
            self._stop_browser(account)
            self._pending_accounts.remove(account)
            self.accounts.remove(account)
            self.account_urls.pop(account, None)
            self.account_labels.pop(account, None)
            with self._profile_lock:
                self.account_profiles.pop(account, None)
            with self.store._connect() as db:
                db.execute("DELETE FROM accounts WHERE id=?", (account,))
            shutil.rmtree(self.store.data_dir / "browser-profiles" / account, ignore_errors=True)

    def _stop_browser(self, account):
        process = self._browser_processes.pop(account, None)
        self._browser_modes.pop(account, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def stop(self):
        self.stop_event.set()
        self.logins.close()
        for account in list(self._browser_processes):
            self._stop_browser(account)
        for worker in self.workers:
            worker.join(timeout=3)
        self._service_lock.close()

    def _worker(self, account):
        while not self.stop_event.is_set():
            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as playwright:
                    if self.logins.is_active(account):
                        self.stop_event.wait(1)
                        continue
                    if account not in self._startup_accounts:
                        self._ensure_managed_browser(account)
                    with self._account_operations[account]:
                        if self.logins.is_active(account):
                            continue
                        ms = ms_provider.connect(playwright, self.account_urls[account])
                    refreshed_at = 0
                    while not self.stop_event.is_set():
                        with self._account_operations[account]:
                            if self.logins.is_active(account):
                                break
                            self.store.verify_identity(account, ms.identity())
                            if time.monotonic() - refreshed_at >= 60:
                                self._refresh_account(ms, account)
                                refreshed_at = time.monotonic()
                            job = self.store.next_for_account(account)
                            if job:
                                if job["status"] == "queued":
                                    self._submit_job(ms, job)
                                elif job["status"] == "download_pending":
                                    if job.get("provider_url"):
                                        self._download_job(job)
                                    else:
                                        self._poll_job(ms, job)
                                else:
                                    self._poll_job(ms, job)
                                current = self.store.get(job["id"])
                                if job["status"] == "queued" or current["status"] in ("completed", "failed"):
                                    refreshed_at = 0  # Refresh after submission or completion.
                        self.stop_event.wait(self.poll_seconds)
            except (Exception, SystemExit) as exc:
                with self._profile_lock:
                    self._account_unavailable_since.setdefault(account, time.time())
                    detail = str(exc) if isinstance(exc, RuntimeError) and "ModelScope" in str(exc) else None
                    if isinstance(exc, ServiceError) and exc.code == "account_identity_mismatch":
                        detail = f"{account} 的浏览器登录身份与原绑定不一致，请重新登录此账号的原 ModelScope 身份。"
                    self.account_profiles[account] = {
                        "id": account, "label": self.account_profiles[account].get("label") or account,
                        "balance": None, "unit": "魔粒",
                        "status": "blocked" if isinstance(exc, ServiceError) else "unavailable",
                        "checked_at": time.time(), "balance_error": None,
                        "session_error": detail or "请检查浏览器登录、本机服务和账号绑定"}
                if isinstance(exc, ServiceError) and exc.code in (
                        "account_identity_mismatch", "account_binding_required", "duplicate_account_identity"):
                    job = self.store.next_for_account(account)
                    if job:
                        self.store.update(job["id"], error=str(exc))
                print(f"WAN worker {self.account_labels[account]}: {type(exc).__name__}", flush=True)
                self.stop_event.wait(self.poll_seconds)

    def bind_accounts(self):
        """Explicit one-time association of pre-fingerprint queues with their logins."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            for account in self.accounts:
                ms = ms_provider.connect(playwright, self.account_urls[account])
                self.store.verify_identity(account, ms.identity(), bind_legacy=True)

    def _submit_job(self, ms, job):
        self.store.update(job["id"], status="uploading", started_at=time.time(), error=None)
        stage = "upload"
        try:
            start_frame = self.store.resolve_data_path(job["first_frame"]) if job["first_frame"] else None
            end_frame = self.store.resolve_data_path(job["last_frame"]) if job["last_frame"] else None
            start_url = ms.upload(start_frame, job["width"], job["height"]) if start_frame else None
            end_url = ms.upload(end_frame, job["width"], job["height"]) if end_frame else None
            stage = "request"
            request = ms.request(MODEL_IDS[job["model"]], start_url, end_url, job["prompt"], job["seconds"],
                                 width=job["width"], height=job["height"], style=job["style"],
                                 negative_prompt=job["negative_prompt"], seed=job["seed"],
                                 **json.loads(job.get("generation_params") or "{}"))
            stage = "estimate"
            self.store.record_event(job["id"], "estimating")
            estimate = ms.estimate(request)
            self.store.update(job["id"], estimate=json.dumps(estimate, ensure_ascii=False))
        except Exception as exc:
            self.store.update(job["id"], status="failed", completed_at=time.time(),
                              failure_details=json.dumps(safe_provider_failure(stage, exc)),
                              error=f"ModelScope {stage} failed; no generation task was submitted.")
            return

        self.store.update(job["id"], status="submitting", error=None)
        try:
            task_id = ms.submit(request)
        except ms_provider.SubmissionRejected as exc:
            self.store.update(job["id"], status="failed", completed_at=time.time(),
                              failure_details=json.dumps(safe_provider_failure("submit", exc)),
                              error="ModelScope rejected the submission.")
            return
        except Exception as exc:
            self.store.update(job["id"], status="unknown_submission",
                              failure_details=json.dumps(safe_provider_failure("submit", exc)),
                              error="Submission may have reached ModelScope; inspect that account before retrying.")
            return
        self.store.update(job["id"], status="in_progress", provider_task_id=str(task_id),
                          submitted_at=time.time(), error=None, failure_details=None)

    def _poll_job(self, ms, job):
        try:
            provider_result = ms.status(job["provider_task_id"])
        except Exception as exc:
            failure = safe_provider_failure("poll", exc)
            if not job.get("failure_details"):
                self.store.record_event(job["id"], "poll_failed", stage="poll")
            self.store.update(job["id"], error="Status check failed; will retry.", failure_details=json.dumps(failure))
            raise RuntimeError("status check failed") from exc
        status = str(provider_result.get("status") or "").upper()
        self.store.update(job["id"], last_polled_at=time.time(), failure_details=None)
        if status != job.get("provider_status") or job.get("failure_details"):
            self.store.record_event(job["id"], "provider_status", provider_status=status[:60])
        if status not in TERMINAL_PROVIDER_STATES:
            self.store.update(job["id"], status="in_progress", provider_status=status or None, error=None)
            return
        if status != "SUCCEED":
            self.store.update(job["id"], status="failed", provider_status=status,
                              completed_at=time.time(), failure_details=json.dumps(safe_provider_failure("generation", payload=provider_result)),
                              error=f"ModelScope task ended with {status}. " + safe_provider_failure("generation", payload=provider_result)["message"])
            return
        video_url = provider_result.get("url") or ms_provider.find_url(provider_result)
        self.store.update(job["id"], status="download_pending", provider_status=status, provider_url=video_url,
                          error=None if video_url else "ModelScope succeeded but returned no video URL; status will be checked again.")
        if video_url:
            self._download_job(self.store.get(job["id"]))

    def _download_job(self, job):
        from .media import save_video_bytes

        output = self.store.jobs_dir / job["id"] / "video.mp4"
        try:
            import requests

            response = requests.get(job["provider_url"], timeout=300)
            response.raise_for_status()
            actual = {}
            save_video_bytes(response.content, output, metadata=actual)
        except Exception as exc:
            attempts = job.get("download_attempts", 0) + 1
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            refresh = status_code in (401, 403) or attempts >= 3
            self.store.update(job["id"], status="download_pending",
                              failure_details=json.dumps(safe_provider_failure("download", exc)),
                              provider_url=None if refresh else job["provider_url"],
                              download_attempts=0 if refresh else attempts,
                              error="Video download failed; will refresh the original task URL." if refresh else
                                    "Video download or validation failed; will retry without generating again.")
            return
        self.store.update(job["id"], status="completed", result_path=self.store.relative_data_path(output),
                          actual_metadata=json.dumps(actual, ensure_ascii=False),
                          completed_at=time.time(), provider_url=None, download_attempts=0, error=None, failure_details=None)


def _error_payload(message, code="invalid_request_error", param=None):
    error = {"message": str(message), "type": code, "param": param, "code": code}
    if getattr(message, "details", None):
        error.update(message.details)
    return {"error": error}


def make_handler(service, token=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "magpie-wan/1.0"

        def log_message(self, fmt, *args):
            # Keep request logs free of headers, query strings and request bodies.
            print(f"WAN HTTP {self.address_string()} {self.command}", flush=True)

        def _authorized(self):
            if token is None:
                return True
            provided = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            return hmac.compare_digest(provided, expected)

        def _json(self, status, body):
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _check_auth(self):
            if not self._authorized():
                self._json(401, _error_payload("A valid Bearer token is required.", "authentication_error"))
                return False
            return True

        def do_GET(self):
            try:
                self._get()
            except ServiceError as exc:
                self._json(exc.status, _error_payload(exc, exc.code))

        def _get(self):
            path = urlsplit(self.path).path
            if path == "/health":
                return self._json(200, {"status": "ok", "workers": len(service.workers),
                                       "service_version": __version__, "protocol_version": PROTOCOL_VERSION})
            if not self._check_auth():
                return
            if path.startswith("/v1/logins/"):
                return self._json(200, service.logins.get(path.rsplit("/", 1)[-1]).snapshot())
            if path == "/v1/models":
                return self._json(200, service.models())
            if path == "/v1/diagnostics":
                return self._json(200, service.diagnostics())
            if path == "/v1/media/roots":
                return self._json(200, {"roots": [str(root) for root in service.media_roots]})
            if path == "/v1/accounts":
                return self._json(200, {"accounts": service.account_snapshot()})
            parts = [unquote(part) for part in path.strip("/").split("/")]
            if len(parts) == 3 and parts[:2] == ["v1", "accounts"]:
                return self._json(200, service.account_snapshot(parts[2]))
            if parts == ["v1", "videos"]:
                query = parse_qs(urlsplit(self.path).query)
                project = (query.get("project") or [None])[0]
                try:
                    limit = max(1, min(200, int((query.get("limit") or [100])[0])))
                    offset = max(0, int((query.get("offset") or [0])[0]))
                except ValueError:
                    raise ServiceError("limit and offset must be whole numbers")
                jobs = service.store.list_jobs(project=project, limit=limit + 1, offset=offset)
                return self._json(200, {"object": "list",
                    "data": [service.public_job(job) for job in jobs[:limit]],
                    "next_offset": offset + limit if len(jobs) > limit else None})
            if len(parts) == 5 and parts[:2] == ["v1", "videos"] and parts[3] == "frames":
                job = service.store.get(parts[2])
                role = parts[4]
                if not job or role not in ("first_frame", "last_frame") or not job.get(role):
                    raise ServiceError("Input frame not found.", 404, "not_found_error")
                frame = service.store.resolve_data_path(job[role])
                if not frame.is_file():
                    raise ServiceError("Input frame is unavailable.", 404, "not_found_error")
                # Re-encode as PNG: never serve source SVG/HTML or embedded active content.
                import io
                with Image.open(frame) as image:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="PNG")
                content = buffer.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            if len(parts) == 3 and parts[:2] == ["v1", "videos"]:
                job = service.store.get(parts[2])
                if job is None:
                    return self._json(404, _error_payload("Video job not found.", "not_found_error"))
                return self._json(200, service.job_details(job))
            if len(parts) == 4 and parts[:2] == ["v1", "videos"] and parts[3] == "content":
                job = service.store.get(parts[2])
                if job is None:
                    return self._json(404, _error_payload("Video job not found.", "not_found_error"))
                if job["status"] != "completed" or not job.get("result_path"):
                    return self._json(409, _error_payload("Video is not ready yet.", "video_not_ready"))
                result_path = service.store.resolve_data_path(job["result_path"])
                if service.store.jobs_dir not in result_path.parents or not result_path.is_file():
                    return self._json(404, _error_payload("Video content is unavailable.", "not_found_error"))
                size = result_path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", f'inline; filename="{parts[2]}.mp4"')
                self.end_headers()
                with result_path.open("rb") as video:
                    shutil.copyfileobj(video, self.wfile)
                return
            self._json(404, _error_payload("Route not found.", "not_found_error"))

        def do_POST(self):
            path = urlsplit(self.path).path
            parts = [unquote(part) for part in path.strip("/").split("/")]
            if path == "/v1/logins" or (len(parts) == 4 and parts[:2] == ["v1", "logins"] and parts[3] == "cancel"):
                if not self._check_auth():
                    return
                # Native clients omit Origin; browser requests must not create login sessions.
                if self.headers.get("Origin"):
                    return self._json(403, _error_payload("Use the Magpie login action."))
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Invalid login request")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("Invalid login request")
                    if path == "/v1/logins":
                        account = body.get("account")
                        if account is not None and (not isinstance(account, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", account)):
                            raise ValueError("Invalid account")
                        return self._json(201, service.logins.create(account))
                    task = service.logins.get(parts[2])
                    task.end("cancelled", "已取消登录。")
                    return self._json(200, task.snapshot())
                except ServiceError as exc:
                    return self._json(exc.status, _error_payload(exc, exc.code))
                except ValueError:
                    return self._json(400, _error_payload("Invalid login request"))
            if path == "/v1/accounts/new":
                return self._json(410, _error_payload("Use /v1/logins to verify login before saving an account."))
            is_submit = path == "/v1/videos"
            is_quote = path == "/v1/videos/estimate"
            is_catalog = path == "/v1/models/search"
            is_validate = path == "/v1/frames/validate"
            is_resolve = len(parts) == 4 and parts[:2] == ["v1", "videos"] and parts[3] == "resolve"
            is_control = len(parts) == 4 and parts[:2] == ["v1", "videos"] and parts[3] == "control"
            if not is_submit and not is_resolve and not is_validate and not is_control and not is_quote and not is_catalog:
                return self._json(404, _error_payload("Route not found.", "not_found_error"))
            if not self._check_auth():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ServiceError(f"request body must be between 1 and {MAX_BODY_BYTES} bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ServiceError("request body must be a JSON object")
                if is_quote or is_catalog:
                    return self._json(200, service.provider_query(body, catalog=is_catalog))
                if is_submit:
                    job, created = service.submit(body, self.headers.get("Idempotency-Key"))
                    result = service.public_job(job)
                    self._json(202 if created else 200, result)
                elif is_validate:
                    self._json(200, service.validate_frames(body))
                else:
                    job = service.control_job(parts[2], {**body, "action": "resolve"} if is_resolve else body)
                    self._json(200, service.job_details(job))
            except json.JSONDecodeError:
                self._json(400, _error_payload("request body must be valid JSON"))
            except ServiceError as exc:
                self._json(exc.status, _error_payload(exc, exc.code))
            except Exception as exc:
                self._json(500, _error_payload("Could not queue video job. Check the local service logs.", "server_error"))

    return Handler


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-accounts", action="store_true",
                        help="Explicitly bind legacy account queues to the configured logins, then exit")
    args = parser.parse_args()
    host = os.environ.get("WAN_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("WAN_SERVICE_PORT", "8765"))
    token = os.environ.get("WAN_SERVICE_TOKEN") or None
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("WAN service accepts local file paths and must bind to loopback")
    roots = parse_media_roots()
    service = WanService(data_dir=default_data_dir(), accounts=parse_accounts(), media_roots=roots,
                         poll_seconds=int(os.environ.get("WAN_SERVICE_POLL_SECONDS", "15")))
    if args.bind_accounts:
        try:
            service.bind_accounts()
            print("Configured account identities are bound.", flush=True)
        finally:
            service.stop()
        return
    httpd = ThreadingHTTPServer((host, port), make_handler(service, token))
    httpd.daemon_threads = True
    service.start()
    print(f"WAN service listening on http://{host}:{port} (accounts: {len(service.accounts)})", flush=True)

    def shutdown(_signum, _frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        service.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
