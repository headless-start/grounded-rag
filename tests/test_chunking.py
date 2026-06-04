from grounded_rag.chunking import Paragraph, _tokens, chunk_paragraphs


def _make_paras(n: int, words_each: int = 120) -> list[Paragraph]:
    return [
        Paragraph(text=" ".join(f"word{i}-{j}" for j in range(words_each)), page=1, paragraph_idx=i)
        for i in range(1, n + 1)
    ]


def test_chunks_respect_max_tokens():
    paras = _make_paras(10)
    chunks = chunk_paragraphs(paras, "doc.md", max_tokens=300, min_tokens=200, overlap_tokens=50)
    assert chunks
    for c in chunks:
        assert len(_tokens(c.text)) <= 300


def test_chunk_ids_are_deterministic():
    paras = _make_paras(5)
    a = chunk_paragraphs(paras, "doc.md")
    b = chunk_paragraphs(paras, "doc.md")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_overlap_carries_tokens_between_chunks():
    paras = _make_paras(8)
    chunks = chunk_paragraphs(paras, "doc.md", max_tokens=250, min_tokens=150, overlap_tokens=60)
    assert len(chunks) >= 2
    # consecutive chunks should share some leading/trailing token overlap
    first_tail = set(_tokens(chunks[0].text)[-60:])
    second_head = set(_tokens(chunks[1].text)[:60])
    assert first_tail & second_head


def test_citation_label_format():
    paras = _make_paras(1)
    chunk = chunk_paragraphs(paras, "data/handbook.pdf")[0]
    label = chunk.citation_label()
    assert label.startswith("[handbook.pdf p1 ")
    assert "¶" in label
