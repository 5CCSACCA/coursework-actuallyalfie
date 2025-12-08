import datetime
from bitnet_service.main import save_llm_completion
from bitnet_service import main

class DummyCollection:
    def __init__(self, inserted):
        self.inserted = inserted

    def insert_one(self, doc):
        self.inserted["mongo_doc"] = doc

class DummyFirestore:
    def __init__(self, inserted):
        self.inserted = inserted

    class _DummyFSCollection:
        def __init__(self, inserted):
            self.inserted = inserted

        def add(self, doc):
            # record that Firestore was written to
            self.inserted["firestore_doc"] = doc

    def collection(self, name):
        return DummyFirestore._DummyFSCollection(self.inserted)

def test_save_llm_completion_inserts_user():
    inserted = {}

    main.app.state.completions = DummyCollection(inserted)
    main.app.state.firestore = DummyFirestore(inserted)

    prompt = "test prompt"
    result = "test output"
    extra = {
        "detection_doc_id": "abc123",
        "objects": ["cat"],
        "user_id": "user-1",
    }

    save_llm_completion(prompt, result, extra)

    doc = inserted["mongo_doc"]
    assert doc["prompt"] == prompt
    assert doc["output"] == result
    assert doc["user_id"] == "user-1"
    assert doc["detection_doc_id"] == "abc123"
    assert doc["objects"] == ["cat"]
    assert isinstance(doc["created_at"], datetime.datetime)
    assert "firestore_doc" in inserted

def test_save_llm_completion_without_user_id():
    inserted = {}

    main.app.state.completions = DummyCollection(inserted)
    main.app.state.firestore = DummyFirestore(inserted)

    prompt = "test"
    result = "output"
    extra = {
        "detection_doc_id": "abc123",
        "objects": ["dog"],
    }

    save_llm_completion(prompt, result, extra)

    doc = inserted["mongo_doc"]
    assert doc["prompt"] == prompt
    assert doc["output"] == result
    assert "firestore_doc" in inserted