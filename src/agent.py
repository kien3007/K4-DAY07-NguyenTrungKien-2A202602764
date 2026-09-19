from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context (numbered chunks for source traceability).
        3. Call the LLM to generate an answer with strict grounding.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Knowledge base is empty. Please add documents before asking questions."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

        # Number each chunk [1], [2], [3] with source traceability
        context_parts = []
        for i, r in enumerate(results, start=1):
            source = r.get("metadata", {}).get("source", r.get("id", f"doc_{i}"))
            context_parts.append(f"[{i}] (Nguồn: {source})\n{r['content']}")
        context = "\n\n".join(context_parts)

        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên các tài liệu được cung cấp.\n"
            "Quy tắc bắt buộc:\n"
            "1. Chỉ sử dụng thông tin có trong phần Ngữ cảnh dưới đây để trả lời.\n"
            "2. Tuyệt đối không bịa đặt hoặc suy đoán thông tin ngoài ngữ cảnh.\n"
            "3. Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là 'Không tìm thấy thông tin trong tài liệu'.\n"
            "4. Trích dẫn rõ nguồn theo số thứ tự tài liệu [1], [2]... khi đưa ra câu trả lời.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)
