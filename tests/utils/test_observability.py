from unittest.mock import Mock

import pytest

from src.utils.observability import ObservabilityManager


class DummyFastAPIInstrumentor:
    def __init__(self):
        self.instrument_app = Mock()


class DummySQLAlchemyInstrumentor:
    def __init__(self):
        self.instrument = Mock()


def test_instrument_fastapi_with_instance(monkeypatch):
    dummy = DummyFastAPIInstrumentor()
    monkeypatch.setattr("src.utils.observability.FastAPIInstrumentor", lambda: dummy)

    obs = ObservabilityManager()
    app = Mock()
    obs.instrument_fastapi(app)

    dummy.instrument_app.assert_called_once_with(app)


def test_instrument_sqlalchemy_with_instance(monkeypatch):
    dummy = DummySQLAlchemyInstrumentor()
    monkeypatch.setattr("src.utils.observability.SQLAlchemyInstrumentor", lambda: dummy)

    obs = ObservabilityManager()
    engine = Mock()
    obs.instrument_sqlalchemy(engine)

    dummy.instrument.assert_called_once_with(engine=engine)
