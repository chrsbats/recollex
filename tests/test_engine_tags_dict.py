import time

from recollex import Recollex


def test_search_result_includes_tags_dict_and_tags_list(tmp_path, monkeypatch):
    # Ensure a clean working directory for any on-disk artifacts
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx"
    rx = Recollex(idx)

    # Add a doc with both dict-style and list-style tags
    now = int(time.time())
    did = rx.add(
        "watch-pipeline unique content 12345",
        tags={"project": "local", "doc_key": "doc.abc123"},
        timestamp=now,
    )

    hits = rx.search("watch-pipeline unique content 12345", k=5)

    assert hits, "Expected at least one search hit"
    hit = hits[0]

    # Base keys
    assert "tags" in hit
    assert "tags_list" in hit
    assert "tags_dict" in hit

    # tags_list should mirror tags
    assert isinstance(hit["tags"], list)
    assert hit["tags_list"] == hit["tags"]

    # tags_dict should contain parsed structured tags
    tags_dict = hit["tags_dict"]
    assert isinstance(tags_dict, dict)
    assert tags_dict.get("project") == "local"
    assert tags_dict.get("doc_key") == "doc.abc123"

    # Ensure doc_id matches the inserted doc
    assert str(did) == hit["doc_id"]


def test_recent_profile_includes_tags_dict_and_project_doc_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx"
    rx = Recollex(idx)

    base_ts = int(time.time())

    # Two docs in different projects
    did_local = rx.add(
        "local project doc",
        tags={"project": "local", "doc_key": "doc.local.1"},
        timestamp=base_ts,
    )
    did_remote = rx.add(
        "remote project doc",
        tags={"project": "remote", "doc_key": "doc.remote.1"},
        timestamp=base_ts + 1,
    )

    # Recent within project="local" should only return the local doc
    hits = rx.search("", profile="recent", project="local", k=5)
    assert hits, "Expected at least one recent hit for project=local"

    hit = hits[0]
    assert hit["doc_id"] == str(did_local)

    # tags_dict should expose project/doc_key
    tags_dict = hit["tags_dict"]
    assert tags_dict.get("project") == "local"
    assert tags_dict.get("doc_key") == "doc.local.1"


def test_search_all_of_tags_accepts_single_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx_allof_single_dict"
    rx = Recollex(idx)

    base_ts = int(time.time())
    did_local = rx.add(
        "doc in local project",
        tags={"project": "local"},
        timestamp=base_ts,
    )
    rx.add(
        "doc in remote project",
        tags={"project": "remote"},
        timestamp=base_ts + 1,
    )

    hits = rx.search("doc in local project", all_of_tags={"project": "local"}, k=10)
    assert hits
    assert hits[0]["doc_id"] == str(did_local)


def test_search_none_of_tags_accepts_single_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx_noneof_single_dict"
    rx = Recollex(idx)

    base_ts = int(time.time())
    did_local = rx.add(
        "doc in local project",
        tags={"project": "local"},
        timestamp=base_ts,
    )
    rx.add(
        "doc in remote project",
        tags={"project": "remote"},
        timestamp=base_ts + 1,
    )

    hits = rx.search("", profile="recent", none_of_tags={"project": "remote"}, k=10)
    assert hits
    assert [h["doc_id"] for h in hits] == [str(did_local)]


def test_search_tag_spec_sequence_formats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx_search_tag_spec_sequence_formats"
    rx = Recollex(idx)

    base_ts = int(time.time())
    did_acme = rx.add(
        "tenant scoped doc",
        tags={"tenant": "acme"},
        timestamp=base_ts,
    )
    rx.add(
        "other tenant doc",
        tags={"tenant": "globex"},
        timestamp=base_ts + 1,
    )

    # all_of_tags accepts tuple form
    hits1 = rx.search("tenant scoped doc", all_of_tags=[("tenant", "acme")], k=10)
    assert hits1
    assert hits1[0]["doc_id"] == str(did_acme)

    # all_of_tags accepts list-of-single-entry-dict form
    hits2 = rx.search("tenant scoped doc", all_of_tags=[{"tenant": "acme"}], k=10)
    assert hits2
    assert hits2[0]["doc_id"] == str(did_acme)

    # none_of_tags accepts tuple form (recent profile with empty query so base is universe)
    hits3 = rx.search("", profile="recent", none_of_tags=[("tenant", "globex")], k=10)
    assert hits3
    assert [h["doc_id"] for h in hits3] == [str(did_acme)]

    # none_of_tags accepts list-of-single-entry-dict form
    hits4 = rx.search("", profile="recent", none_of_tags=[{"tenant": "globex"}], k=10)
    assert hits4
    assert [h["doc_id"] for h in hits4] == [str(did_acme)]


def test_search_one_of_tags_accepts_single_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx_search_oneof_single_dict"
    rx = Recollex(idx)

    base_ts = int(time.time())
    did_local = rx.add(
        "local doc",
        tags={"project": "local"},
        timestamp=base_ts,
    )
    rx.add(
        "remote doc",
        tags={"project": "remote"},
        timestamp=base_ts + 1,
    )

    hits = rx.search("local doc", one_of_tags={"project": "local"}, k=10)
    assert hits
    assert hits[0]["doc_id"] == str(did_local)


def test_search_everything_sentinel_in_tag_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    idx = tmp_path / "idx_search_everything_sentinel"
    rx = Recollex(idx)

    base_ts = int(time.time())
    did_local = rx.add(
        "local doc",
        tags={"project": "local"},
        timestamp=base_ts,
    )
    did_remote = rx.add(
        "remote doc",
        tags={"project": "remote"},
        timestamp=base_ts + 1,
    )

    # "everything" in an all_of_tags list means "no restriction" for that list.
    hits = rx.search("", profile="recent", all_of_tags=["everything"], k=10)
    assert [h["doc_id"] for h in hits] == [str(did_remote), str(did_local)]

    # "everything" in none_of_tags means "do not exclude anything"
    hits2 = rx.search("", profile="recent", none_of_tags=["everything"], k=10)
    assert [h["doc_id"] for h in hits2] == [str(did_remote), str(did_local)]
