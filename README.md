# 🚚 Shipping Assistant (RAG + LangChain + Output Parser)

An Arabic (Egyptian dialect) customer-support assistant for shipping and courier questions. It runs **100% free** on an open-source Hugging Face model, with **no API keys required**.

Ask it anything about shipping (tracking, delays, damaged parcels, customs, COD, packaging, returns, etc.) and it answers from a small FAQ knowledge base, falling back to general knowledge when needed. If a question is off-topic, it politely declines.

## Features

- **Retrieval-Augmented Generation (RAG):** each FAQ entry is embedded and stored in FAISS; the top 3 most relevant entries are passed to the model as context.
- **Domain filter:** a fast keyword check plus an LLM yes/no check keeps the assistant on topic (shipping only).
- **Ticket classification with an Output Parser:** every in-scope message is turned into a structured support ticket using LangChain's `PydanticOutputParser`:
  - `category`: shipping, payment, returns, account, technical, other
  - `priority`: low, medium, high
  - `sentiment`: positive, neutral, negative
  - `summary`: one-sentence summary of the issue
  - `needs_human`: whether a human agent should step in
- **Safe fallback:** if the model returns invalid JSON, the ticket defaults to `other / medium / neutral / needs_human = True`.
- **Anti-hallucination prompt:** the assistant is told not to invent company-specific prices, timings, or numbers, and to point users to the company's official site instead.
- **Gradio UI:** an interactive web interface showing the answer, the generated ticket, and the retrieved sources.

## How It Works

```
User message
     │
     ▼
Domain filter (keywords + LLM yes/no)
     │ off-topic ──► polite refusal
     ▼
Ticket classifier (LLM + PydanticOutputParser) ──► structured ticket (JSON)
     │
     ▼
Retriever (FAISS + multilingual embeddings, top 3)
     │
     ▼
Answer chain (LLM + retrieved context) ──► final answer
```

## Tech Stack

| Component | Choice |
|---|---|
| LLM | `Qwen/Qwen2.5-3B-Instruct` (optional: `Qwen2.5-7B-Instruct` in 4-bit) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (supports Arabic) |
| Vector store | FAISS (CPU) |
| Orchestration | LangChain |
| Output parsing | Pydantic + `PydanticOutputParser` |
| UI | Gradio |

## Getting Started

### Run on Kaggle (recommended)

1. Upload `shipping_assistant_kaggle.ipynb` to [Kaggle](https://www.kaggle.com) as a new notebook.
2. In the notebook **Settings** (right sidebar):
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Internet:** On
3. Click **Run All**.
4. The last cell launches a Gradio app and prints a public share link.

### Dependencies

Installed automatically by the first notebook cell:

```
langchain langchain-core langchain-community langchain-huggingface
langchain-text-splitters sentence-transformers faiss-cpu pydantic
gradio accelerate bitsandbytes
```

A GPU is needed to run the model at a reasonable speed.

## Knowledge Base

The FAQ is a plain text block in the notebook, saved to `/kaggle/working/data/faq.txt`. Each entry is a question and answer separated by a blank line:

```
السؤال: ...
الجواب: ...
```

The included entries are **generic examples**. To use it for a real company, replace them with that company's actual policies, prices, and working hours.

## Example Usage

```python
result = handle_message("ايه الفرق بين الشحن البحري والجوي؟")

print(result["answer"])   # the assistant's reply
print(result["ticket"])   # structured ticket (dict), or None if off-topic
print(result["sources"])  # FAQ entries used as context
```

Off-topic questions (for example, a cooking question) return a polite refusal and no ticket.

## Configuration

| What | Where |
|---|---|
| Switch to a larger model | Uncomment the 7B block in section 3 |
| Use your own fine-tuned model | Change `MODEL_ID` in section 3 |
| Retrieve more/fewer FAQ entries | Change `k` in `search_kwargs` in section 6 |
| Response length limits | `domain_llm`, `ticket_llm`, `answer_llm` token limits in section 3 |
| Shipping keywords for the domain filter | `SHIPPING_KEYWORDS` in section 7 |

## Limitations

- The model is small, so answers can occasionally be inaccurate. Always verify company-specific details.
- The sample FAQ contains example information only, not real company data.
- Optimized for Arabic (Egyptian dialect); other languages may work but are not the focus.
- Ticket classification depends on the model producing valid JSON; invalid output falls back to a default ticket.

## Project Structure

```
.
├── shipping_assistant_kaggle.ipynb   # full pipeline: data, model, chains, UI
└── README.md
```

## License

Add a license of your choice (for example MIT) before publishing.
