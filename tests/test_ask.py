"""第1回の単発質問・継続会話を、実APIを呼ばずに検証する。"""

from copy import deepcopy

import pytest
from google import genai
from google.genai import errors, types

from app import ask


def response(text="テスト回答", *, usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]
        ))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=7
        ) if usage else None,
    )


@pytest.fixture
def sdk(monkeypatch):
    # キーは実在しないテスト専用値。通信箇所を置き換える。
    client = genai.Client(api_key="test-only-not-a-real-key")
    calls = []

    def generate(**kwargs):
        calls.append(deepcopy(kwargs))
        return response()

    monkeypatch.setattr(client.models, "generate_content", generate)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-only-not-a-real-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    yield client, calls
    client.close()


def test_single_question_keeps_return_value_and_reports_tokens(sdk, capsys):
    _, calls = sdk
    assert ask.ask("LoRAとは？") == "テスト回答"
    assert calls[0]["contents"] == "LoRAとは？"
    assert calls[0]["model"] == ask.MODEL
    assert "入力: 12 / 出力: 7" in capsys.readouterr().err


def test_chat_sends_previous_user_and_model_turns(sdk, monkeypatch, capsys):
    _, calls = sdk
    answers = iter(["それを短く説明して", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    ask.chat("LoRAとは？")
    assert len(calls) == 2
    contents = calls[1]["contents"]
    assert [item.role for item in contents] == ["user", "model", "user"]
    assert [item.parts[0].text for item in contents] == [
        "LoRAとは？", "テスト回答", "それを短く説明して"
    ]
    assert capsys.readouterr().err.count("入力: 12 / 出力: 7") == 2


def test_chat_skips_empty_input_and_quits_without_request(sdk, monkeypatch):
    _, calls = sdk
    answers = iter(["   ", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    ask.chat()
    assert calls == []


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_chat_exits_cleanly_on_terminal_end(sdk, monkeypatch, exception):
    _, calls = sdk

    def stop(_):
        raise exception

    monkeypatch.setattr("builtins.input", stop)
    ask.chat()
    assert calls == []


def test_missing_usage_is_reported_as_unknown(sdk, monkeypatch, capsys):
    client, _ = sdk
    monkeypatch.setattr(client.models, "generate_content", lambda **kwargs: response(usage=False))
    assert ask.ask("質問") == "テスト回答"
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().err


def test_zero_tokens_are_not_reported_as_unknown(capsys):
    result = response()
    result.usage_metadata.prompt_token_count = 0
    result.usage_metadata.candidates_token_count = 0
    ask.print_usage(result)
    assert "入力: 0 / 出力: 0" in capsys.readouterr().err


def test_chat_api_error_is_explained_and_exits(sdk, monkeypatch, capsys):
    client, _ = sdk

    def fail(**kwargs):
        raise errors.ClientError(429, {"error": {"message": "test rate limit"}})

    monkeypatch.setattr(client.models, "generate_content", fail)
    monkeypatch.setattr("sys.argv", ["ask", "--chat", "質問"])
    with pytest.raises(SystemExit) as exc:
        ask.main()
    assert exc.value.code == 1
    assert "レート制限" in capsys.readouterr().out
