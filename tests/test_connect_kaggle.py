import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def test_connection_requests_only_read_only_notebook_scope(monkeypatch, capsys):
    calls = []
    class Client:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None
    class OAuth:
        def __init__(self, client):
            assert isinstance(client, Client)
        def authenticate(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'kagglesdk', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'kagglesdk.kaggle_client', SimpleNamespace(KaggleClient=Client))
    monkeypatch.setitem(sys.modules, 'kagglesdk.kaggle_oauth', SimpleNamespace(KaggleOAuth=OAuth))
    path = Path(__file__).parents[1] / 'scripts/connect-kaggle.py'
    spec = importlib.util.spec_from_file_location('connect_kaggle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    assert calls == [{'scopes': ['kernels.get:*']}]
    assert 'Never paste credentials into chat' in capsys.readouterr().out
