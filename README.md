**English** | [Deutsch](README.de.md)

# Hamburg AI Assistant

A retrieval-augmented (RAG) Telegram bot for a student community in Hamburg. It turns years of unstructured group-chat conversations into a searchable knowledge base and answers new questions in Farsi, German and English, using only what the community has already shared.

Practical questions about residence permits, banks, health insurance, housing or university were asked and answered again and again in the group chat and were almost impossible to find later. This project extracts those answers, structures them and makes them searchable.

**Status:** test phase (September 2026). The bot is not used by the community yet.

**What is not in this repository:** the knowledge base and the original chat data. They contain content from real conversations and are not published. To try the pipeline anyway, the repository includes a small fictional dataset in `examples/`.

## How it works

```text
Telegram export (HTML)
        |
        v
  parse_telegram.py      one JSON record per message (sender, reply_to, timestamp, ...)
        |
        v
  build_clusters.py      group messages into conversations (time gaps + reply chains)
        |
        v
  extract_qa.py          LLM extracts anonymised question/answer pairs per conversation
        |
        v
  merge and deduplicate  qa_master_deduped.json
        |
        v
  categorize_qa.py       keyword rules assign one of 11 topic categories
        |
        v
  build_index.py         multilingual sentence embeddings + FAISS index
        |
        v
  query_kb.py            retrieve top entries, generate the answer with GPT-4o-mini
        |
        v
  bot.py                 Telegram bot (python-telegram-bot)
```

The current knowledge base has 227 entries in 11 categories, built from about 16,600 messages of an 18-file export (June 2021 to March 2026).

## Design decisions

- **Multilingual embeddings.** The knowledge base mixes Farsi, German and English, so a keyword search would miss most matches. `paraphrase-multilingual-mpnet-base-v2` puts similar questions close together regardless of the language they are written in. It runs locally, so there is no per-query embedding cost.
- **Answers only from the knowledge base.** The system prompt tells the model to answer from the retrieved entries and not from its own knowledge. If nothing similar enough is found (cosine similarity below 0.35), the bot says so instead of guessing.
- **Top 6 instead of top 1.** On this dataset the similarity scores sit in a narrow band, so the best match is not always at rank 1. The model sees six candidates and weighs them. It can also point out that a procedure changed over time when entries from different years disagree.
- **Explicit language detection.** Even with an instruction in the prompt, the model sometimes answered in the wrong language. The bot now detects the language of the question (a script check for Farsi, `langdetect` for German and English) and names it in the prompt.
- **Uncertain and outdated information is marked.** The extraction step records a `note` for rumours and time-bound rules such as old Covid regulations, and the answer step passes these notes on.
- **Anonymisation.** The extraction prompt forbids sender names and the output has no sender fields. Institutions, doctors and products may be named because that is useful information. The prompt is an instruction, not a guarantee, which is one reason the knowledge base stays private.
- **Group behaviour.** In a group chat the bot only answers when it is mentioned or when someone replies to it. In a private chat it answers everything.

## Project layout

```text
scripts/
  parse_telegram.py         Telegram HTML export -> JSONL
  build_clusters.py         messages -> conversation clusters
  extract_qa.py             clusters -> Q&A pairs (OpenAI)
  categorize_qa.py          keyword-based topic categories
  build_index.py            embeddings + FAISS index
  query_kb.py               retrieval and answer generation, also a command-line tool
  bot.py                    Telegram bot
  diagnose_retrieval.py     shows the full similarity ranking for one question
  deploy.sh                 runs on the server: install dependencies, restart the service
tests/
  test_categorize.py        unit tests for the categorisation rules
  test_build_clusters.py    unit tests for the clustering
  eval_bot.py               end-to-end evaluation with an LLM as judge
examples/
  sample_knowledge_base.json   small fictional dataset for trying the pipeline
.github/workflows/
  ci.yml                    lint, tests, JSON validation, secret scan
  eval.yml                  evaluation (manual or weekly)
  cd.yml                    manual deployment to the server
.pre-commit-config.yaml     local checks before each commit
```

## Try it with the sample data

You need Python 3.12 and an OpenAI API key. The first run downloads the embedding model (about 1 GB).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY (and a bot token for bot.py)

python scripts/build_index.py examples/sample_knowledge_base.json -o kb_index/
python scripts/query_kb.py "How do I pay in the canteen?"
```

`query_kb.py` without a question starts an interactive mode. To run the Telegram bot, put `TELEGRAM_BOT_TOKEN` in `.env` and start `python scripts/bot.py`.

The sample entries are invented (fictional city, library and bank) and only exist to show the format.

## Using your own chat export

```bash
python scripts/parse_telegram.py messages1.html messages2.html -o messages.jsonl
python scripts/build_clusters.py messages.jsonl -o clusters.jsonl --gap 45
python scripts/extract_qa.py clusters.jsonl -o qa_dataset.json
# merge and deduplicate the pairs into knowledge_base/qa_master_deduped.json
# (this step is done outside the scripts in this repository)
python scripts/categorize_qa.py knowledge_base/qa_master_deduped.json -o knowledge_base/qa_categorized.json
python scripts/build_index.py knowledge_base/qa_master_deduped.json -o kb_index/
```

To see why a question does or does not find an entry, `diagnose_retrieval.py` prints the ranking of all entries and can highlight one of them:

```bash
python scripts/diagnose_retrieval.py "your question" --highlight m04
```

## Evaluation

`tests/eval_bot.py` measures the pipeline instead of relying on a feeling that it works:

1. **Known-answer check.** It takes two answered entries per category (up to 22 questions), asks the bot the stored question and lets `gpt-4o-mini` judge whether the bot's answer is consistent with the stored answer in substance, not in wording.
2. **Refusal check.** It asks five questions that have nothing to do with the knowledge base (bicycle repair, baking, films, ...) and checks that the bot says it does not know instead of inventing an answer. For a bot that talks about visa deadlines, a confident wrong answer is worse than no answer.

In the latest run, 95 % of the 22 reference questions were answered consistently with the stored answer, and all five off-topic questions were refused.

These numbers need context. The reference questions come from the knowledge base itself, so the test shows that retrieval and generation work together end to end, but it says little about how the bot copes with new phrasings from real users. A judge model is also not a human. The planned pilot with community members will show how it does on real questions.

## Tests and CI/CD

```bash
pip install pytest
pytest tests/
```

| Workflow | When | What it does |
| --- | --- | --- |
| `ci.yml` | every push and pull request to `main` | byte-compiles the code, runs `ruff`, runs the unit tests, validates the knowledge base JSON (skipped if it is not present) and scans the git history for secrets with `gitleaks` |
| `eval.yml` | manually and every Monday | builds the index and runs the evaluation. It costs OpenAI credits, so it is kept apart from `ci.yml`. It needs the private knowledge base and skips itself without it |
| `cd.yml` | manually, after typing `deploy` | connects to the server over SSH and runs `scripts/deploy.sh` |

`deploy.sh` runs on the server: it activates the virtual environment, installs the dependencies and restarts the `systemd` service, then prints the service status. The workflow needs the repository secrets `VPS_HOST`, `VPS_USER` and `VPS_SSH_KEY`, so it only works in a repository that has them.

The bot runs as a `systemd` service on a VPS. The server setup (key-only SSH login, firewall, fail2ban, automatic security updates) is done on the server and is not part of this repository.

The pre-commit hooks (`pip install pre-commit && pre-commit install`) run the same `ruff` rules and a few file checks locally before each commit.

## Limitations

- **Test phase.** The bot has not been used by real users yet, so its quality on real questions is unknown.
- **Static knowledge.** The knowledge base does not update itself. New chat messages have to go through the pipeline again.
- **Outdated answers.** Some information in the chats gets old. Time-bound entries are marked, but the marking is only as good as the extraction step.
- **Crude categories.** The categories come from keyword rules on the topic field. For example, a topic containing "Anmeldung" ends up under housing even if it is about a language course.
- **Blocking OpenAI call.** The bot calls OpenAI synchronously, which blocks its event loop while it waits. That is fine for occasional questions in a small group. For more traffic it should switch to the async client.
- **Pinned for Linux.** `requirements.txt` is a full freeze from the Linux server, using CPU-only PyTorch. On other systems some pins may need adjusting.
- **Model dependence.** Extraction, answers and judging all use `gpt-4o-mini`. Results can differ with other models.

## Privacy

The knowledge base is built from real conversations of a private community, and the raw export contains sender names. Keep the exports, the intermediate JSONL files, the knowledge base and the search index out of git, and do not commit `.env` files. The `.gitignore` in this repository excludes them. The sample data contains only invented content.
