"""第1回: 単発質問と会話履歴を使った連続質問。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # /exit または /quit で終了
    uv run python -m app.ask --chat "LoRAとは何ですか？"

単発質問は会話履歴を持たない。--chat ではSDKのチャットで履歴を引き継ぐ。
履歴はこの実行中のメモリだけに保持し、ファイルには保存しない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import sys

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response: types.GenerateContentResponse) -> None:
    """APIがトークン数を返さない場合も、0と誤認しないよう表示する。"""
    usage = response.usage_metadata
    prompt = usage.prompt_token_count if usage is not None else None
    output = usage.candidates_token_count if usage is not None else None
    print(
        f"[tokens] 入力: {prompt if prompt is not None else '不明'} / "
        f"出力: {output if output is not None else '不明'}",
        file=sys.stderr,
    )


def ask(question: str) -> str:
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text or "[テキストの回答がありませんでした]"


def chat(first_question: str | None = None) -> None:
    """同じチャットを使い続け、直前までの質問と回答を次の質問に渡す。"""
    print("[会話] /exit または /quit で終了。履歴は終了時に破棄します。")
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        question = first_question
        while True:
            if question is None:
                try:
                    question = input("あなた> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            question = question.strip()
            if question in ("/exit", "/quit"):
                break
            if question:
                print(f"[質問] {question}")
                response = session.send_message(question)
                print(response.text or "[テキストの回答がありませんでした]")
                print_usage(response)
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(args.question)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
