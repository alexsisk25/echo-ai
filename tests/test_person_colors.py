"""Per-person chip hues: numbers only, persistence, auto reset."""

import pytest
from fastapi.testclient import TestClient

from app import config, db, main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "colors.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    return TestClient(main.app)


def test_hue_choice_persists_and_overwrites(client):
    assert client.get("/api/people/colors").json() == {}
    r = client.put("/api/people/Felix Vivanco/color", json={"color": 178})
    assert r.status_code == 200
    assert client.get("/api/people/colors").json() == {"Felix Vivanco": 178}
    # Any hue on the wheel, not just preset values; floats round.
    client.put("/api/people/Felix Vivanco/color", json={"color": 203.6})
    assert client.get("/api/people/colors").json() == {"Felix Vivanco": 204}


def test_auto_and_null_reset_to_automatic(client):
    client.put("/api/people/Ana/color", json={"color": 25})
    client.put("/api/people/Ana/color", json={"color": None})
    assert client.get("/api/people/colors").json() == {}
    client.put("/api/people/Ana/color", json={"color": 25})
    client.put("/api/people/Ana/color", json={"color": "auto"})
    assert client.get("/api/people/colors").json() == {}


def test_only_hue_numbers_are_accepted(client):
    # Raw color values and junk never enter: hex, names, out-of-range.
    assert client.put("/api/people/Ana/color",
                      json={"color": "#ff0000"}).status_code == 422
    assert client.put("/api/people/Ana/color",
                      json={"color": "chartreuse"}).status_code == 422
    assert client.put("/api/people/Ana/color",
                      json={"color": -5}).status_code == 422
    assert client.put("/api/people/Ana/color",
                      json={"color": 360}).status_code == 422
    assert client.get("/api/people/colors").json() == {}


def test_colors_route_is_not_shadowed_by_the_person_route(client):
    # /api/people/colors must never be read as a person named "colors".
    assert client.get("/api/people/colors").status_code == 200


def test_two_people_may_share_a_hue(client):
    client.put("/api/people/Ana/color", json={"color": 178})
    client.put("/api/people/Bob/color", json={"color": 178})
    assert client.get("/api/people/colors").json() == {"Ana": 178, "Bob": 178}


def test_legacy_preset_name_rows_migrate_to_hues(client, tmp_path,
                                                 monkeypatch):
    # The brief preset-name era: a raw 'teal' row must come back as its
    # historical hue after the next connect, and never break the GET.
    conn = db.connect()
    conn.execute("INSERT INTO person_colors (name, color) VALUES "
                 "('Old Row', 'teal')")
    conn.execute("INSERT INTO person_colors (name, color) VALUES "
                 "('Junk Row', 'not-a-color')")
    conn.commit()
    conn.close()
    db.connect().close()  # migration runs on connect
    body = client.get("/api/people/colors").json()
    assert body["Old Row"] == 178
    assert "Junk Row" not in body
