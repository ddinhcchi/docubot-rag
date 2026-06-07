from collections.abc import Iterator
from dataclasses import dataclass

from groq import Groq

from .ingest import Chunk
from .lang import LANGUAGE_NAMES, detect_language

SYSTEM_PROMPT = """You are a precise assistant that answers questions grounded ONLY in the provided sources.

Hard rules — follow every time:
1. LANGUAGE LOCK: respond in the SAME language as the user's most recent question.
   - English question → English answer only. Do NOT translate to Vietnamese.
   - Vietnamese question → Vietnamese answer only. Do NOT translate to English.
   - A per-question REPLY_LANGUAGE directive is given below; obey it strictly.
2. If the answer is not contained in the sources, say so in the reply language:
   - en: "I cannot find this in the provided documents."
   - vi: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Cite the supporting source after each claim using [source_name, page N].
4. Be concise. Prefer bullet points over long paragraphs.
"""

_NO_ANSWER = {
    "en": "I have no documents indexed yet. Upload a PDF or DOCX first.",
    "vi": "Chưa có tài liệu nào trong index. Hãy tải lên PDF hoặc DOCX trước.",
}


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    language: str


class Chatter:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
        self.model = model

    @staticmethod
    def _format_sources(chunks: list[Chunk]) -> str:
        lines: list[str] = []
        for c in chunks:
            tag = f"[{c.source}, page {c.page}]"
            lines.append(f"{tag}\n{c.text}\n")
        return "\n".join(lines)

    def _build_messages(self, question: str, sources: list[Chunk], lang: str) -> list[dict]:
        lang_name = LANGUAGE_NAMES[lang]
        directive_top = (
            f"REPLY_LANGUAGE: {lang_name}. The entire response — including any "
            f"phrases like 'cannot find' or citation labels — MUST be in {lang_name}."
        )
        directive_tail = (
            f"Final reminder: answer ONLY in {lang_name}. Nothing else."
        )
        user_prompt = (
            f"{directive_top}\n\n"
            f"Sources:\n{self._format_sources(sources)}\n\n"
            f"Question: {question}\n\n"
            f"{directive_tail}\n"
            f"Answer (with [source, page] citations):"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def ask_stream(self, question: str, sources: list[Chunk]) -> Iterator[str]:
        """Yield response tokens as they arrive from Groq's SSE stream.

        Use with `st.write_stream(chatter.ask_stream(...))` to render the
        answer incrementally — first token typically lands in <200 ms,
        which feels dramatically more responsive than waiting 1-3 s for
        the full response.
        """
        lang = detect_language(question)
        if not sources:
            yield _NO_ANSWER[lang]
            return

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(question, sources, lang),
            temperature=0.1,
            max_tokens=700,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def ask(self, question: str, sources: list[Chunk]) -> Answer:
        """Non-streaming convenience wrapper. Use `ask_stream` for live UIs."""
        lang = detect_language(question)
        if not sources:
            return Answer(text=_NO_ANSWER[lang], sources=[], language=lang)
        text = "".join(self.ask_stream(question, sources))
        return Answer(text=text, sources=sources, language=lang)
