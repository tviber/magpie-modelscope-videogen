import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from magpie_wan.login import LoginTask


def test_native_login_task_status_isolated_and_terminal():
    manager = SimpleNamespace(service=Mock())
    task = LoginTask(manager, 'a', True, 30)
    other = LoginTask(manager, 'b', True, 30)
    task.state = 'waiting'
    assert task.id != other.id
    assert not hasattr(task, 'server')
    assert not hasattr(task, 'url')
    snapshot = task.snapshot()
    assert set(snapshot) == {'id', 'account', 'state', 'error', 'expires_at'}
    task.end('cancelled', 'cancelled')
    task.end('completed')
    assert task.state == 'cancelled'
    assert other.state == 'starting'


def test_expired_login_cleans_pending_account_without_connecting(monkeypatch):
    service = Mock()
    service.store.account_for_identity.return_value = None
    service.stop_event = threading.Event()
    service.account_urls = {'a': 'http://127.0.0.1:1'}
    task = LoginTask(SimpleNamespace(service=service), 'a', True, -1)
    try:
        task.run()
        assert task.state == 'expired'
        service.discard_pending_account.assert_called_once_with('a')
        service.activate_browser_account.assert_not_called()
    finally:
        task.end("cancelled")


def test_handoff_preserves_session_cookies_and_checks_identity_twice(monkeypatch):
    from magpie_wan.service import ServiceError
    service = Mock()
    service.store.account_for_identity.return_value = None
    service.stop_event = threading.Event()
    service._browser_processes = {'a': Mock()}
    task = LoginTask(SimpleNamespace(service=service), 'a', True, 60)
    original, restored, foreground, background = Mock(), Mock(), Mock(), Mock()
    cookies = [{'name': 'session', 'value': 'fixture', 'domain': 'modelscope.cn', 'path': '/'}]
    original.pg.context.cookies.return_value = cookies
    background.contexts = [Mock()]
    imported = []
    background.contexts[0].add_cookies.side_effect = lambda values: imported.extend(dict(v) for v in values)
    monkeypatch.setattr(task, 'connect', lambda _: background)
    monkeypatch.setattr(task, 'authenticated', lambda _: restored)
    try:
        task.handoff(Mock(), foreground, original)
        assert task.state == 'completed'
        foreground.new_browser_cdp_session.return_value.send.assert_called_once_with('Browser.close')
        service._ensure_managed_browser.assert_called_once_with('a', headless=True)
        assert imported[0]['value'] == 'fixture'
        assert cookies == []
        assert service._refresh_account.call_args_list == [
            ((original, 'a'),), ((restored, 'a'),)]
        service.activate_browser_account.assert_called_once_with('a')
    finally:
        task.end("cancelled")


def test_headless_identity_mismatch_never_activates_account(monkeypatch):
    from magpie_wan.service import ServiceError
    service = Mock()
    service.store.account_for_identity.return_value = None
    service.stop_event = threading.Event()
    service._browser_processes = {}
    service._refresh_account.side_effect = [None, ServiceError('identity mismatch', 409)]
    task = LoginTask(SimpleNamespace(service=service), 'a', True, 60)
    original, background = Mock(), Mock()
    original.pg.context.cookies.return_value = []
    background.contexts = [Mock()]
    monkeypatch.setattr(task, 'connect', lambda _: background)
    monkeypatch.setattr(task, 'authenticated', lambda _: Mock())
    try:
        with pytest.raises(ServiceError, match='identity mismatch'):
            task.handoff(Mock(), Mock(), original)
        assert task.state != 'completed'
        service.activate_browser_account.assert_not_called()
    finally:
        task.end("cancelled")


def test_repeat_login_refreshes_original_account_without_moving_jobs(monkeypatch):
    from magpie_wan import provider
    service = Mock()
    service.stop_event = threading.Event()
    service.account_urls = {'old': 'http://127.0.0.1:9223', 'new': 'http://127.0.0.1:9224'}
    service._startup_accounts = {'old'}
    service._account_operations = {'old': threading.Lock()}
    service.store.account_for_identity.return_value = 'old'
    manager = SimpleNamespace(service=service, is_active=lambda _: False)
    task = LoginTask(manager, 'new', True, 60)
    ms = Mock()
    ms.identity.return_value = 'Id:123'
    ms.pg.context.cookies.return_value = [
        {'domain': '.modelscope.cn', 'name': 'session', 'value': 'fixture'},
        {'domain': '.other.cn', 'name': 'unrelated', 'value': 'fixture'}]
    page = Mock(url='https://modelscope.cn/')
    pw = Mock()
    pw.chromium.connect_over_cdp.return_value.contexts = [SimpleNamespace(pages=[page])]
    restored = Mock()
    monkeypatch.setattr(provider, 'MS', Mock(return_value=restored))
    task.handoff(pw, Mock(), ms)
    assert task.state == 'completed'
    assert task.snapshot()['account'] == 'new'  # Correlation remains unchanged.
    assert task.snapshot()['result_account'] == 'old'
    assert service.store.verify_identity.call_count == 2
    service._refresh_account.assert_called_once_with(restored, 'old')
    service.discard_pending_account.assert_called_once_with('new')
    service.activate_browser_account.assert_not_called()
    assert len(page.context.add_cookies.call_args.args[0]) == 1
    assert service._account_operations['old'].acquire(blocking=False)
    service._account_operations['old'].release()


def test_repeat_login_identity_failure_does_not_touch_existing_cookies(monkeypatch):
    from magpie_wan.service import ServiceError
    service = Mock()
    service.stop_event = threading.Event()
    service.account_urls = {'old': 'http://127.0.0.1:9223'}
    service._account_operations = {'old': threading.Lock()}
    service.store.account_for_identity.return_value = 'old'
    service.store.verify_identity.side_effect = ServiceError('identity mismatch', 409)
    task = LoginTask(SimpleNamespace(service=service, is_active=lambda _: False), 'new', True, 60)
    pw = Mock()
    with pytest.raises(ServiceError, match='identity mismatch'):
        task.handoff(pw, Mock(), Mock())
    pw.chromium.connect_over_cdp.assert_not_called()
    assert task.state != 'completed'
