"""Semantic search: local embeddings in sqlite-vec, plus an LLM answer
pass for cross-meeting questions.

Embeddings are computed on this machine with a small sentence-transformers
model and never leave it. Only the retrieved transcript excerpts (plain
text) go to the cloud LLM when answering a question. The model choice is
committed: changing it means re-embedding everything.
"""

import json

from app import config

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on whitespace boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def index_transcript(conn, transcript_id: int, full_text: str) -> int:
    """(Re)index one transcript. Returns the number of chunks stored."""
    import sqlite_vec

    old = conn.execute(
        "SELECT id FROM chunks WHERE transcript_id = ?", (transcript_id,)
    ).fetchall()
    for (chunk_id,) in old:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE transcript_id = ?", (transcript_id,))

    chunks = chunk_text(full_text)
    if chunks:
        vectors = _get_model().encode(chunks, normalize_embeddings=True)
        for text, vec in zip(chunks, vectors):
            cur = conn.execute(
                "INSERT INTO chunks (transcript_id, text) VALUES (?, ?)",
                (transcript_id, text),
            )
            conn.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(vec)),
            )
    conn.commit()
    return len(chunks)


# The vector index picks its k nearest chunks across the whole archive
# and the folder filter is applied afterwards, so a folder-scoped search
# has to ask the index for more than it needs. A fixed window is not
# enough: on a big archive a small folder's chunks can sit outside it
# entirely, and the search would come back empty while the folder really
# does hold matching text. So the window widens until enough of the
# folder's own chunks survive, or until the whole index has been read.
FOLDER_OVERFETCH = 8


def folder_is_indexed(conn, folder_id: int) -> bool:
    """Whether this folder has any embedded chunks at all. Separates
    "nothing in here is indexed" from "nothing here matched"."""
    row = conn.execute(
        """
        SELECT 1 FROM chunks c
        JOIN transcripts t ON t.id = c.transcript_id
        JOIN recordings r ON r.id = t.recording_id
        WHERE r.folder_id = ? LIMIT 1
        """,
        (folder_id,),
    ).fetchone()
    return row is not None


def search(conn, query: str, k: int = 8,
           folder_id: int | None = None) -> list[dict]:
    """Nearest transcript chunks to the query, with recording context.
    Pass folder_id to search only the recordings in that folder."""
    import sqlite_vec

    vec = _get_model().encode([query], normalize_embeddings=True)[0]
    clause = "" if folder_id is None else " AND r.folder_id = ?"
    sql = f"""
        SELECT c.text, c.transcript_id, t.recording_id, s.summary_json,
               v.distance, r.privileged
        FROM chunks_vec v
        JOIN chunks c ON c.id = v.rowid
        JOIN transcripts t ON t.id = c.transcript_id
        JOIN recordings r ON r.id = t.recording_id
        LEFT JOIN summaries s ON s.transcript_id = t.id
        WHERE v.embedding MATCH ? AND v.k = ?{clause}
        ORDER BY v.distance
        """

    def run(fetch):
        params = [sqlite_vec.serialize_float32(vec), fetch]
        if folder_id is not None:
            params.append(folder_id)
        return conn.execute(sql, params).fetchall()

    if folder_id is None:
        rows = run(k)
    else:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        fetch = min(max(k * FOLDER_OVERFETCH, 40), max(total, 1))
        rows = run(fetch)
        while len(rows) < k and fetch < total:
            fetch = min(fetch * 4, total)
            rows = run(fetch)
    results = []
    for text, tid, rec_id, summary_json, distance, privileged in rows:
        summary = json.loads(summary_json) if summary_json else {}
        results.append({
            "text": text,
            "transcript_id": tid,
            "recording_id": rec_id,
            "title": summary.get("title"),
            "date": summary.get("date"),
            "distance": distance,
            "privileged": bool(privileged),
        })
    return results[:k]


ASK_PROMPT = """You answer questions about the user's recorded meetings using only the transcript excerpts below. If the excerpts do not contain the answer, say so plainly. Cite which meeting(s) the answer comes from by their bracketed number. Keep the answer to a few sentences.

Question: {question}

Excerpts:
{excerpts}"""


def ask(conn, question: str, k: int = 8,
        folder_id: int | None = None) -> dict:
    """Answer a cross-meeting question from retrieved excerpts. Pass
    folder_id to answer from one folder only.

    Privacy vault rule: text from privileged recordings never reaches
    the cloud. If any retrieved excerpt is privileged, the answer is
    synthesized by the local model; if the local model is unavailable,
    privileged excerpts are dropped and the cloud answers from the rest.
    A privileged FOLDER is stricter: without the local model there is no
    answer at all, because there is nothing left it would be safe to ask.
    """
    from app import llm, summarize

    folder_privileged = False
    if folder_id is not None:
        from app import digest
        folder_privileged = digest.folder_is_privileged(conn, folder_id)

    hits = search(conn, question, k=k, folder_id=folder_id)
    if not hits:
        if folder_id is None:
            answer = "No transcripts are indexed yet."
        elif folder_is_indexed(conn, folder_id):
            answer = "Nothing in this folder matched that question."
        else:
            answer = "Nothing in this folder is indexed yet."
        return {"answer": answer, "sources": []}

    note = None
    any_privileged = any(h["privileged"] for h in hits) or folder_privileged
    privileged_call = any_privileged
    if any_privileged:
        try:
            llm._get_local()
        except llm.LocalModelUnavailable:
            if folder_privileged:
                return {"answer": (
                    "This folder is marked privileged, so its text never "
                    "goes to the cloud, and the local model is not "
                    "installed. Install it to ask questions here."),
                    "sources": []}
            hits = [h for h in hits if not h["privileged"]]
            privileged_call = False
            note = ("Some privileged recordings matched but were excluded: "
                    "the local model is not installed, and privileged "
                    "content never goes to the cloud.")
            if not hits:
                return {"answer": note, "sources": []}

    seen = {}
    for h in hits:
        seen.setdefault(h["recording_id"], h)
    sources = list(seen.values())
    numbered = {h["recording_id"]: i + 1 for i, h in enumerate(sources)}

    excerpts = "\n\n".join(
        f"[{numbered[h['recording_id']]}] {h['title'] or 'Untitled'}"
        f"{' (' + h['date'] + ')' if h['date'] else ''}:\n{h['text']}"
        for h in hits
    )
    answer = summarize._call_llm(
        ASK_PROMPT.format(question=question, excerpts=excerpts),
        privileged=privileged_call,
    )
    if note:
        answer = f"{answer.strip()}\n\n({note})"
    return {
        "answer": answer.strip(),
        "sources": [
            {"recording_id": h["recording_id"], "title": h["title"],
             "n": numbered[h["recording_id"]]}
            for h in sources
        ],
    }
