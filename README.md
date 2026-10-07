# SmartHR AI Assistant using RAG

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG) chat assistant** that
answers HR-related questions using two knowledge sources:

- An **Employee Handbook (PDF)** — unstructured text covering company policies
  (leave, probation, remote work, etc.)
- An **Employee Database (CSV)** — structured records (ID, name, department,
  manager, position, office, performance rating, etc.)

The assistant keeps **conversational memory** per session, so it can correctly
resolve follow-up questions (e.g., "Who is the manager of Employee 10?" →
"Which department does **he** work in?").

---

## Why This Design

The project needed to combine **structured** and **unstructured** knowledge
sources with conversational memory. The chosen architecture makes this
concrete rather than theoretical:

1. The PDF and CSV are loaded through genuinely different code paths
   (`PyPDFLoader` vs. `pandas` + manual `Document` construction), then merged
   into a single FAISS index so retrieval doesn't need to know which source
   holds the answer.
2. Memory is handled by LangChain's own `RunnableWithMessageHistory` +
   `create_history_aware_retriever`, which lets the LLM rewrite ambiguous
   follow-up questions into standalone search queries — a more robust
   approach than manually concatenating chat history into every prompt.
3. A lightweight **router** intercepts greetings/identity questions before
   they ever reach the LLM or retriever, keeping the RAG pipeline focused on
   real HR questions only.
4. The LLM runs **locally** (Qwen2.5-1.5B-Instruct via Hugging Face
   Transformers) instead of calling an external API, which avoids rate
   limits/quotas entirely and keeps the whole pipeline self-contained.

---

## Architecture

```
PDF ──► PyPDFLoader ──┐
                       ├──► RecursiveCharacterTextSplitter ──► HuggingFaceEmbeddings ──► FAISS Index
CSV ──► pandas ──► Documents ──┘

User Question ──► route_question() ──► greeting/identity? ──► canned reply (no LLM call)
                        │
                   real HR question
                        │
                        ▼
          history_aware_retriever ──► FAISS retrieval ──► create_stuff_documents_chain
                                                                    │
                                                        Local Qwen2.5-1.5B-Instruct
                                                                    │
                                                                Final Answer
                                                                    │
                                                     RunnableWithMessageHistory
                                              (stores turn in InMemoryChatMessageHistory)
```

Full details, including a step-by-step component table and design decisions,
are included as a Markdown cell at the **end of the notebook**
(`SmartHR_AI_Assistant_using_RAG___Mid_Term_FIXED.ipynb`).

---

## How the Assistant Works, Step by Step

1. **Data Integration**: The PDF is loaded page by page; the CSV is loaded row
   by row and converted into readable employee profile text blocks.
2. **Chunking**: All documents (from both sources) are split into ~500-character
   overlapping chunks.
3. **Embedding & Indexing**: Each chunk is embedded with
   `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.
4. **Routing**: Every incoming message is checked against greeting/identity
   patterns first; only genuine HR questions proceed to retrieval.
5. **History-aware retrieval**: The LLM rewrites the question using prior chat
   history (if needed), then the top 3 relevant chunks are retrieved from FAISS.
6. **Generation**: The retrieved chunks + chat history + the question are
   combined into a prompt and sent to the local Qwen model.
7. **Memory**: Each turn is stored in a session-specific
   `InMemoryChatMessageHistory`, so later questions can reference earlier ones.
8. **Testing**: Memory tests and a batch of demo questions verify retrieval
   accuracy and correct follow-up handling.

---

## Files in This Project

| File | Purpose |
|---|---|
| `SmartHR_AI_Assistant_using_RAG___Mid_Term_FIXED.ipynb` | Main notebook — full working code |
| `employee_handbook_v2.pdf` | HR policy source (unstructured) — supply your own |
| `employees_v2.csv` | Employee records source (structured) — supply your own |
| `README.md` (this file) | High-level project summary |

---

## Setup Before Running

1. Upload `employee_handbook_v2.pdf` and `employees_v2.csv` to the Colab
   session (or Google Drive) — the notebook expects them by these filenames.
2. Run all cells top to bottom. The local LLM (`Qwen2.5-1.5B-Instruct`) downloads
   automatically on first run (~3GB) — no API key required.
3. Use the interactive chat loop at the end, or the pre-built test cells, to
   try it out.

---

## 📍 Where to Put This File

Put `README.md` **in the same folder/submission as the notebook** — i.e.
alongside `SmartHR_AI_Assistant_using_RAG___Mid_Term_FIXED.ipynb` (and the
PDF/CSV files), not inside the notebook itself. If you're submitting on:

- **Google Drive / a zipped folder**: just drop `README.md` in the same folder
  as the `.ipynb` file before zipping/uploading.
- **GitHub**: put it at the **root of the repository** — GitHub automatically
  displays a file named exactly `README.md` on the repo's main page.
- **An LMS (like the one for this course)**: upload it as a separate
  attachment next to the notebook submission, so instructors can read the
  project summary without opening the whole notebook first.
