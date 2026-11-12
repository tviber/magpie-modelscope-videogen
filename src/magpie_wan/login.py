"""Visible official-site login, native Magpie completion, then headless handoff.

The callback never accepts credentials or a client assertion of success. Only
provider identity verification in the owned browser can complete a task.
"""
import secrets
import threading
import time

from . import provider

TERMINAL = {"completed", "failed", "cancelled", "expired"}


class LoginTask:
    def __init__(self, manager, account, fresh, timeout):
        self.manager, self.service = manager, manager.service
        self.id = secrets.token_hex(24)
        self.account, self.fresh = account, fresh
        self.deadline = time.monotonic() + timeout
        self.expires_at = time.time() + timeout
        self.lock = threading.RLock()
        self.state, self.error = "starting", None
        self.result_account = None
        self.thread = threading.Thread(target=self.run, name="wan-login", daemon=True)

    def snapshot(self):
        with self.lock:
            result = {"id": self.id, "account": self.account, "state": self.state,
                      "error": self.error, "expires_at": self.expires_at}
            if self.state == "completed" and self.result_account:
                result["result_account"] = self.result_account
            return result

    def end(self, state, error=None):
        with self.lock:
            if self.state in TERMINAL:
                return
            self.state, self.error = state, error

    def live(self):
        if self.service.stop_event.is_set():
            self.end("cancelled", "服务已停止，请重新登录。")
        if time.monotonic() >= self.deadline:
            self.end("expired", "登录已超时，请回到 Magpie 重试。")
        return self.state not in TERMINAL

    def connect(self, pw):
        while self.live():
            try:
                return pw.chromium.connect_over_cdp(self.service.account_urls[self.account], timeout=2000)
            except Exception:
                time.sleep(0.25)
        return None

    def authenticated(self, browser):
        pages = [p for c in browser.contexts for p in c.pages if not p.is_closed() and provider.trusted_url(p.url)]
        for page in reversed(pages):
            if not self.live():
                return None
            try:
                return provider.MS(page, self.service.account_urls[self.account])
            except Exception:
                pass  # The real page may still be awaiting SMS or a slider challenge.
        return None

    def handoff(self, pw, browser, ms):
        if self.fresh:
            existing = self.service.store.account_for_identity(ms.identity())
            if existing and existing != self.account:
                self.reconnect_existing(pw, ms, existing)
                return
        # Bind/check before switching; a different login must never replace the account.
        self.service._refresh_account(ms, self.account)
        with self.lock:
            if not self.live():
                return
            self.state = "switching"
        context = ms.pg.context
        # Session cookies need not survive a Chrome restart. Carry them in memory
        # into the same private profile, then verify the identity again headlessly.
        cookies = context.cookies()
        try:
            session = browser.new_browser_cdp_session()
            session.send("Browser.close")
        except Exception:
            pass  # Chrome may close the transport before acknowledging Browser.close.
        process = self.service._browser_processes.get(self.account)
        if process is not None:
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        if not self.live():
            return
        self.service._ensure_managed_browser(self.account, headless=True)
        background = self.connect(pw)
        if background is None:
            return
        try:
            background.contexts[0].add_cookies(cookies)
        finally:
            cookies.clear()
        # New page creation/navigation and API availability can lag behind CDP readiness.
        for _ in range(15):
            if not self.live():
                return
            verified = self.authenticated(background)
            if verified:
                self.service._refresh_account(verified, self.account)
                with self.lock:
                    if not self.live():
                        return
                    self.service.activate_browser_account(self.account)
                    self.end("completed")
                return
            time.sleep(1)
        self.end("failed", "登录信息未能在后台浏览器中恢复，请重新登录。")

    def reconnect_existing(self, pw, ms, account):
        """Refresh the original session at a worker boundary; never move its jobs."""
        from .service import ServiceError
        if account not in self.service.account_urls:
            raise ServiceError(f"此登录对应已有账号 {account}，请恢复该账号的浏览器配置后重试。", 409)
        operation = self.service._account_operations[account]
        while self.live():
            if operation.acquire(timeout=0.25):
                break
        else:
            return
        cookies = []
        try:
            if not self.live():
                return
            if self.manager.is_active(account):
                raise ServiceError("这个账号正在另一窗口登录，请先完成该窗口。", 409)
            # Check the new login twice, around cookie export, before touching the old session.
            self.service.store.verify_identity(account, ms.identity())
            cookies = ms.pg.context.cookies()
            self.service.store.verify_identity(account, ms.identity())
            if account not in self.service._startup_accounts:
                self.service._ensure_managed_browser(account)
            try:
                browser = pw.chromium.connect_over_cdp(self.service.account_urls[account], timeout=5000)
            except Exception as exc:
                raise ServiceError(f"已识别为原账号 {account}，但原浏览器未运行。请打开原账号浏览器后重试。", 409) from exc
            pages = [p for c in browser.contexts for p in c.pages if provider.trusted_url(p.url)]
            if not pages:
                raise ServiceError("原账号浏览器需要保留 ModelScope 官网页面。", 409)
            page = pages[0]
            if not self.live():
                return
            # Only ModelScope cookies are needed by this adapter's official API requests.
            page.context.add_cookies([c for c in cookies if c.get("domain", "").lstrip(".") in provider.MODELSCOPE_HOSTS])
            restored = provider.MS(page, self.service.account_urls[account])
            self.service._refresh_account(restored, account)
            with self.lock:
                if not self.live():
                    return
                self.service.discard_pending_account(self.account)
                self.result_account = account
                self.end("completed")
        finally:
            cookies.clear()
            operation.release()

    def run(self):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                # Only the login thread switches modes, after its task has been
                # registered so workers cannot race the visible browser launch.
                if not self.live():
                    return
                self.service._ensure_managed_browser(self.account, headless=False)
                browser = self.connect(pw)
                if browser is None:
                    return
                with self.lock:
                    if not self.live():
                        return
                    self.state = "waiting"
                while self.live():
                    if not browser.is_connected() or not any(c.pages for c in browser.contexts):
                        self.end("cancelled", "登录浏览器已关闭，请回到 Magpie 重试。")
                        return
                    ms = self.authenticated(browser)
                    if ms:
                        self.handoff(pw, browser, ms)
                        return
                    time.sleep(1)
        except Exception as exc:
            from .service import ServiceError
            message = str(exc) if isinstance(exc, ServiceError) else "登录连接中断，请回到 Magpie 重试。"
            self.end("failed", message)
        finally:
            if self.state not in TERMINAL:
                self.end("cancelled", "服务已停止，请重新登录。")
            if self.state != "completed":
                if self.fresh:
                    self.service.discard_pending_account(self.account)
                else:
                    # Keep saved credentials, but do not leave a visible login window behind.
                    self.service._stop_browser(self.account)


class LoginManager:
    def __init__(self, service):
        self.service = service
        self.tasks = {}
        self.lock = threading.RLock()

    def is_active(self, account):
        # Tasks are only inserted/removed on the management thread under the GIL;
        # copy before iterating, without taking the task lock from a service lock.
        return any(t.account == account and (t.state not in TERMINAL or t.thread.is_alive()) for t in list(self.tasks.values()))

    def create(self, account=None, timeout=540):
        from .service import ServiceError
        with self.lock:
            # Bound retained terminal records and concurrent browser launches.
            for key, old in list(self.tasks.items()):
                if old.state in TERMINAL and time.time() > old.expires_at + 120:
                    self.tasks.pop(key)
            if sum(t.state not in TERMINAL for t in self.tasks.values()) >= 4:
                raise ServiceError("已有多个登录窗口，请先完成或取消。", 409)
            if account:
                if account not in self.service.account_urls or account in self.service._startup_accounts:
                    raise ServiceError("此账号不是托管账号，请使用浏览器账号同步入口。", 400)
                if self.is_active(account):
                    raise ServiceError("这个账号已有登录任务。", 409)
                if self.service.store.active_counts([account]).get(account, 0):
                    raise ServiceError("账号有未完成的视频任务，请完成任务后再重新登录。", 409)
            else:
                account = self.service.launch_browser_account(pending=True, headless=False)["id"]
            fresh = account in self.service._pending_accounts
            try:
                with self.service._account_operations[account]:
                    if self.service.store.active_counts([account]).get(account, 0):
                        raise ServiceError("账号有未完成的视频任务，请稍后重新登录。", 409)
                    task = LoginTask(self, account, fresh, timeout)
                    self.tasks[task.id] = task
                    task.thread.start()
                    return task.snapshot()
            except Exception:
                if fresh:
                    self.service.discard_pending_account(account)
                raise

    def get(self, task_id):
        from .service import ServiceError
        with self.lock:
            task = self.tasks.get(task_id)
        if task is None:
            raise ServiceError("登录任务不存在或已过期。", 404)
        return task

    def close(self):
        for task in list(self.tasks.values()):
            task.end("cancelled", "服务已停止。")
        for task in list(self.tasks.values()):
            task.thread.join(timeout=5)
