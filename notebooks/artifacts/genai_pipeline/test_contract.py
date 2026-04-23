from app_fastapi import app


def test_health_contract():
    # Contract-level smoke test placeholder
    assert app.title == "genai-rag-service"
