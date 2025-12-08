import importlib
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import prometheus_client

def get_yolo_main():
    class DummyCounter:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

    prometheus_client.Counter = DummyCounter

    import yolo_service.main as main
    importlib.reload(main)
    return main

def test_verify_firebase_token_missing():
    main = get_yolo_main()
    with pytest.raises(HTTPException) as exc:
        main.verify_firebase_token(None)

    assert exc.value.status_code == 401

def test_verify_firebase_token_invalid(monkeypatch):
    main = get_yolo_main()

    def fake_verify(token):
        raise ValueError("bad token")

    monkeypatch.setattr(main.firebase_auth, "verify_id_token", fake_verify)

    creds = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="invalid",
    )

    with pytest.raises(HTTPException):
        main.verify_firebase_token(creds)

def test_verify_firebase_token_valid(monkeypatch):
    main = get_yolo_main()

    def fake_verify(token):
        return {"uid": "user-123"}

    monkeypatch.setattr(main.firebase_auth, "verify_id_token", fake_verify)

    creds = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="valid-token",
    )

    decoded = main.verify_firebase_token(creds)
    assert decoded["uid"] == "user-123"