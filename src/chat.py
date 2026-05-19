from dataclasses import dataclass

from groq import Groq

from .ingest import Chunk

SYSTEM_PROMPT = """You are a precise assistant that answers questions grounded ONLY in the provided sources.

Rules:
- If the answer is not contained in the sources, reply: "I cannot find this in the provided documents."
- Cite the supporting source after each claim using the format [source_name, page N].
- Reply in the same language the user asked in (e.g. Vietnamese question → Vietnamese answer).
- Be concise. Prefer bullet points over long paragraphs.
"""


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


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

    def ask(self, question: str, sources: list[Chunk]) -> Answer:
        if not sources:
            return Answer(
                text="I have no documents indexed yet. Upload a PDF or DOCX first.",
                sources=[],
            )
        user_prompt = (
            f"Sources:\n{self._format_sources(sources)}\n\n"
            f"Question: {question}\n\n"
            "Answer (with citations):"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return Answer(
            text=resp.choices[0].message.content or "",
            sources=sources,
        )
