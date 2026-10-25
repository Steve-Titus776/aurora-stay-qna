"""Retrieve property table rows with their labels alongside semantic results."""

import re


def _property_code(text):
    match = re.search(r"ASH[-\s]+([A-Z0-9]+)", text, re.IGNORECASE)
    return "ASH-" + match.group(1).upper() if match else None


def _property_rows(vectorstore):
    """Recover labeled property records from the indexed Markdown chunks."""
    from langchain_core.documents import Document

    state = {}
    records = []
    for _, doc_id in sorted(vectorstore.index_to_docstore_id.items()):
        document = vectorstore.docstore.search(doc_id)
        if not isinstance(document, Document):
            continue
        source = document.metadata.get("source", "")
        context = state.setdefault(source, {"headers": None, "note": "", "section": ""})
        for line in document.page_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                context.update(headers=None, note="", section=line.lstrip("# "))
            elif not line.startswith("|") and re.search(r"\b(verif\w*|training|live profile)\b", line, re.I):
                context["note"] = line
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells[0].lower() == "property":
                context["headers"] = cells
                continue
            headers = context["headers"]
            code = _property_code(cells[0])
            if not code or not headers or len(cells) != len(headers):
                continue
            fields = dict(zip(headers, cells))
            content = "\n".join(f"{label}: {value}" for label, value in fields.items())
            if context["note"]:
                content += "\nReference guidance: " + context["note"]
            records.append((code, fields, Document(
                page_content=content,
                metadata={**document.metadata, "section": context["section"],
                          "retrieval_method": "property_table_lookup"},
            )))
    return records


def retrieve_policy_documents(query, k, vectorstore):
    """Prioritize a requested property's labeled row for property amenities/fees."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer, such as 2 or 3.")

    semantic = vectorstore.similarity_search(query, k=k)
    topics = [pattern for words, pattern in (
        (r"\bpets?\b", r"\bpet"),
        (r"\b(parking|valet|garage)\b", r"parking"),
        (r"\b(breakfast|breakfasts)\b", r"breakfast"),
        (r"\bcheck[ -]?(in|out)\b", r"check"),
    ) if re.search(words, query, re.I)]
    if not topics:
        return semantic

    records = _property_rows(vectorstore)
    requested = list(dict.fromkeys("ASH-" + code.upper() for code in
                                  re.findall(r"ASH[-\s]+([A-Z0-9]+)", query, re.I)))
    if not requested:
        # Also recognize a city/property name when the guest omits its code.
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        for code, fields, _ in records:
            name = next(iter(fields.values()))
            name = re.sub(r"ASH[-\s]+[A-Z0-9]+", "", name, flags=re.I)
            name_words = {word for word in re.findall(r"[a-z]+", name.lower()) if len(word) > 2}
            if len(name_words & query_words) >= 2 and code not in requested:
                requested.append(code)

    selected = []
    for code in requested:
        matches = [(fields, document) for record_code, fields, document in records
                   if record_code == code and
                   all(any(re.search(topic, label, re.I) for label in fields) for topic in topics)]
        if matches:
            # Prefer the fuller property matrix (e.g. including the pet-count limit).
            _, document = max(matches, key=lambda match: len(match[0]))
            selected.append(document)
    seen = set()
    results = []
    for document in selected + semantic:
        if document.page_content not in seen:
            seen.add(document.page_content)
            results.append(document)
        if len(results) == k:
            break
    return results
