from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

pytest.importorskip("google.genai")

_KEYS = ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ARENA_PROMPT_LLM", "ARENA_DATA_DIR")


def _answer(text: str) -> tuple[int, dict]:
    return 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": 9}}


_ERROR = (500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}})


class _Model(http.server.BaseHTTPRequestHandler):
    replies: list[tuple[int, dict]] = []
    requests: list[dict] = []

    def do_POST(self) -> None:
        cls = type(self)
        cls.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        status, doc = cls.replies.pop(0) if len(cls.replies) > 1 else cls.replies[0]
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        del args


@pytest.fixture
def model() -> Iterator[tuple[str, type[_Model]]]:
    _Model.replies, _Model.requests = [_answer("pong")], []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _Model
    finally:
        server.shutdown()
        server.server_close()


def _python(code: str, url: str, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    return subprocess.run([sys.executable, *(args or ("-c", textwrap.dedent(code)))], cwd=cwd, env={**env, "LLM_API_ENDPOINT": url}, capture_output=True, text=True, timeout=120, check=False)


_ASK = """
from arena_simulation_setup.generative.scenario.propose.client import Unavailable, ask
try:
    reply = ask('be brief', 'ping', model='m')
except Unavailable as exc:
    print('unavailable:', exc)
else:
    print(reply.text, reply.usage['input_tokens'], reply.usage['total_tokens'])
"""


def test_ask_retries_once_after_an_api_error(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    url, endpoint = model
    endpoint.replies = [_ERROR, _answer("pong")]
    done = _python(_ASK, url, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "pong 7 9"
    assert len(endpoint.requests) == 2
    assert endpoint.requests[1]["systemInstruction"]["parts"] == [{"text": "be brief"}]


def test_ask_gives_up_after_the_second_api_error(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    url, endpoint = model
    endpoint.replies = [_ERROR]
    done = _python(_ASK, url, tmp_path)
    assert done.stdout.startswith("unavailable: the model call failed: ServerError"), done.stdout + done.stderr
    assert len(endpoint.requests) == 2


def test_an_empty_answer_is_unavailable(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    url, endpoint = model
    endpoint.replies = [_answer("")]
    done = _python(_ASK, url, tmp_path)
    assert done.stdout.strip() == "unavailable: the model returned an empty response", done.stdout + done.stderr


def test_a_call_asks_on_a_miss_writes_the_cache_and_hits_it_next(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    url, endpoint = model
    endpoint.replies = [_answer('```json\n{"role": "work"}\n```')]
    code = """
    import json
    from pathlib import Path
    from arena_simulation_setup.generative.scenario.promptgen.llm import call
    first = call('s', 'u', cache_dir=Path('cache'), model='m')
    (entry,) = Path('cache').iterdir()
    stored = json.loads(entry.read_text())
    second = call('s', 'u', cache_dir=Path('cache'), model='m')
    print(json.dumps([first.cached, second.cached, first.json(), second.text == first.text, second.sha256 == first.sha256, stored['usage']['input_tokens'], stored['model']]))
    """
    done = _python(code, url, tmp_path)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [False, True, {"role": "work"}, True, True, 7, "m"]
    assert len(endpoint.requests) == 1


def _world(tmp_path: Path) -> Path:
    root = tmp_path / "worlds" / "w"
    root.mkdir(parents=True)
    zones = [{"name": name, "polygon": [[10.0 * i, 0.0], [10.0 * i + 4, 0.0], [10.0 * i + 4, 4.0], [10.0 * i, 4.0]], "floor": False} for i, name in enumerate(("top_room_0", "main_hallway"))]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    return root


def test_classify_tail_asks_about_the_named_zones(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    url, endpoint = model
    endpoint.replies = [_answer('{"top_room_0": {"role": "work", "why": "a desk"}}')]
    code = """
    from arena_simulation_setup.generative.scenario.propose.zone_roles import classify_tail
    zones = [{"name": "top_room_0", "corners": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}], "entities": {"static": []}}]
    proposals, complaints = classify_tail(zones, world="w", names=["top_room_0"], model="m")
    print([(p.zone, p.role.value, p.why, p.model) for p in proposals], complaints)
    """
    done = _python(code, url, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[('top_room_0', 'work', 'a desk', 'm')] []"
    [request] = endpoint.requests
    assert "top_room_0" in request["contents"][0]["parts"][0]["text"]


def test_classify_write_stores_the_answers_in_the_world_kg(model: tuple[str, type[_Model]], tmp_path: Path) -> None:
    from arena_simulation_setup.kg import load

    url, endpoint = model
    endpoint.replies = [_answer('{"top_room_0": {"role": "work", "why": "a desk"}}')]
    root = _world(tmp_path)
    done = _python("", url, tmp_path, "-m", "arena_simulation_setup.generative.scenario.cli", "classify", "w", "--worlds", str(root.parent), "--write", "--model", "m")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 fact(s) written" in done.stdout
    facts = [(f.subject, str(f.object), f.params["by"], f.params["model"]) for f in load(root / "kg.yaml").facts if f.verb == "has_purpose"]
    assert facts == [("room:top_room_0", "purpose:work", "llm", "m")]
    assert len(endpoint.requests) == 1
