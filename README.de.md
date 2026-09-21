[English](README.md) | **Deutsch**

# Hamburg AI Assistant

Ein RAG-basierter Telegram-Bot (Retrieval-Augmented Generation) für eine Studierendencommunity in Hamburg. Er verwandelt jahrelange unstrukturierte Gruppenchats in eine durchsuchbare Wissensbasis und beantwortet neue Fragen auf Farsi, Deutsch und Englisch, und zwar ausschließlich mit dem, was die Community bereits geteilt hat.

Praktische Fragen zu Aufenthaltstiteln, Banken, Krankenversicherung, Wohnen oder Studium wurden im Gruppenchat immer wieder neu gestellt und beantwortet und waren später kaum wiederzufinden. Dieses Projekt extrahiert diese Antworten, strukturiert sie und macht sie durchsuchbar.

**Status:** Testphase (September 2026). Der Bot wird von der Community noch nicht genutzt.

**Was nicht in diesem Repository liegt:** die Wissensbasis und die ursprünglichen Chatdaten. Sie enthalten Inhalte aus echten Gesprächen und werden nicht veröffentlicht. Um die Pipeline trotzdem auszuprobieren, enthält das Repository unter `examples/` einen kleinen fiktiven Datensatz.

## Funktionsweise

```text
Telegram-Export (HTML)
        |
        v
  parse_telegram.py      ein JSON-Datensatz pro Nachricht (Absender, reply_to, Zeitstempel, ...)
        |
        v
  build_clusters.py      Nachrichten zu Gesprächen gruppieren (Zeitabstände + Antwortketten)
        |
        v
  extract_qa.py          LLM extrahiert anonymisierte Frage-Antwort-Paare je Gespräch
        |
        v
  Zusammenführen und deduplizieren   qa_master_deduped.json
        |
        v
  categorize_qa.py       Schlagwortregeln ordnen eine von 11 Themenkategorien zu
        |
        v
  build_index.py         multilinguale Sentence-Embeddings + FAISS-Index
        |
        v
  query_kb.py            passende Einträge abrufen, Antwort mit GPT-4o-mini erzeugen
        |
        v
  bot.py                 Telegram-Bot (python-telegram-bot)
```

Die aktuelle Wissensbasis umfasst 227 Einträge in 11 Kategorien, erstellt aus rund 16.600 Nachrichten eines Exports mit 18 Dateien (Juni 2021 bis März 2026).

## Designentscheidungen

- **Multilinguale Embeddings.** Die Wissensbasis mischt Farsi, Deutsch und Englisch, eine Stichwortsuche würde die meisten Treffer verpassen. `paraphrase-multilingual-mpnet-base-v2` bringt ähnliche Fragen unabhängig von der Sprache nah zusammen. Das Modell läuft lokal, es fallen also keine Kosten pro Abfrage an.
- **Antworten nur aus der Wissensbasis.** Der System-Prompt weist das Modell an, aus den abgerufenen Einträgen zu antworten und nicht aus eigenem Wissen. Wird nichts ausreichend Ähnliches gefunden (Kosinus-Ähnlichkeit unter 0,35), sagt der Bot das, statt zu raten.
- **Top 6 statt Top 1.** In diesem Datensatz liegen die Ähnlichkeitswerte eng beieinander, der beste Treffer steht nicht immer auf Rang 1. Das Modell sieht sechs Kandidaten und wägt sie ab. Es kann auch darauf hinweisen, dass sich ein Verfahren mit der Zeit geändert hat, wenn Einträge aus verschiedenen Jahren sich widersprechen.
- **Explizite Spracherkennung.** Trotz Anweisung im Prompt antwortete das Modell gelegentlich in der falschen Sprache. Der Bot erkennt deshalb die Sprache der Frage (Skriptprüfung für Farsi, `langdetect` für Deutsch und Englisch) und nennt sie im Prompt.
- **Unsichere und veraltete Informationen werden markiert.** Die Extraktion vermerkt in einem `note`-Feld Gerüchte und zeitgebundene Regeln wie alte Corona-Vorgaben, und die Antwortgenerierung gibt diese Hinweise weiter.
- **Anonymisierung.** Der Extraktions-Prompt verbietet Absendernamen, und die Ausgabe enthält keine Absenderfelder. Institutionen, Ärzte und Produkte dürfen genannt werden, weil das nützliche Information ist. Der Prompt ist eine Anweisung und keine Garantie, auch deshalb bleibt die Wissensbasis privat.
- **Verhalten in Gruppen.** In einem Gruppenchat antwortet der Bot nur, wenn er erwähnt wird oder jemand auf ihn antwortet. Im privaten Chat antwortet er auf alles.

## Projektstruktur

```text
scripts/
  parse_telegram.py         Telegram-HTML-Export -> JSONL
  build_clusters.py         Nachrichten -> Gesprächscluster
  extract_qa.py             Cluster -> Q&A-Paare (OpenAI)
  categorize_qa.py          schlagwortbasierte Themenkategorien
  build_index.py            Embeddings + FAISS-Index
  query_kb.py               Abruf und Antwortgenerierung, auch als Kommandozeilenwerkzeug
  bot.py                    Telegram-Bot
  diagnose_retrieval.py     zeigt das vollständige Ähnlichkeitsranking für eine Frage
  deploy.sh                 läuft auf dem Server: Abhängigkeiten installieren, Dienst neu starten
tests/
  test_categorize.py        Unit-Tests für die Kategorisierungsregeln
  test_build_clusters.py    Unit-Tests für das Clustering
  eval_bot.py               End-to-End-Evaluation mit einem LLM als Bewerter
examples/
  sample_knowledge_base.json   kleiner fiktiver Datensatz zum Ausprobieren der Pipeline
.github/workflows/
  ci.yml                    Lint, Tests, JSON-Validierung, Secret-Scan
  eval.yml                  Evaluation (manuell oder wöchentlich)
  cd.yml                    manuelles Deployment auf den Server
.pre-commit-config.yaml     lokale Prüfungen vor jedem Commit
```

## Mit den Beispieldaten ausprobieren

Benötigt werden Python 3.12 und ein OpenAI-API-Schlüssel. Beim ersten Start wird das Embedding-Modell heruntergeladen (rund 1 GB).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # OPENAI_API_KEY eintragen (und für bot.py einen Bot-Token)

python scripts/build_index.py examples/sample_knowledge_base.json -o kb_index/
python scripts/query_kb.py "How do I pay in the canteen?"
```

`query_kb.py` ohne Frage startet einen interaktiven Modus. Für den Telegram-Bot `TELEGRAM_BOT_TOKEN` in `.env` eintragen und `python scripts/bot.py` starten.

Die Beispieleinträge sind erfunden (fiktive Stadt, Bibliothek und Bank) und zeigen nur das Format.

## Mit einem eigenen Chat-Export

```bash
python scripts/parse_telegram.py messages1.html messages2.html -o messages.jsonl
python scripts/build_clusters.py messages.jsonl -o clusters.jsonl --gap 45
python scripts/extract_qa.py clusters.jsonl -o qa_dataset.json
# Paare zu knowledge_base/qa_master_deduped.json zusammenführen und deduplizieren
# (dieser Schritt geschieht außerhalb der Skripte in diesem Repository)
python scripts/categorize_qa.py knowledge_base/qa_master_deduped.json -o knowledge_base/qa_categorized.json
python scripts/build_index.py knowledge_base/qa_master_deduped.json -o kb_index/
```

Um zu sehen, warum eine Frage einen Eintrag findet oder nicht, gibt `diagnose_retrieval.py` das Ranking aller Einträge aus und kann einen davon hervorheben:

```bash
python scripts/diagnose_retrieval.py "deine Frage" --highlight m04
```

## Evaluation

`tests/eval_bot.py` misst die Pipeline, statt sich auf ein Bauchgefühl zu verlassen:

1. **Test mit bekannten Antworten.** Aus jeder Kategorie werden zwei beantwortete Einträge gezogen (bis zu 22 Fragen). Der Bot bekommt die gespeicherte Frage, und `gpt-4o-mini` beurteilt, ob die Bot-Antwort inhaltlich, nicht im Wortlaut, zur gespeicherten Antwort passt.
2. **Ablehnungstest.** Fünf Fragen, die nichts mit der Wissensbasis zu tun haben (Fahrradreparatur, Backen, Filme, ...), prüfen, ob der Bot sagt, dass er es nicht weiß, statt eine Antwort zu erfinden. Bei einem Bot, der über Visa-Fristen spricht, ist eine selbstsichere falsche Antwort schlimmer als keine.

Im letzten Lauf wurden 95 % der 22 Referenzfragen inhaltlich passend zur gespeicherten Antwort beantwortet, und alle fünf themenfremden Fragen wurden abgelehnt.

Diese Zahlen brauchen Kontext. Die Referenzfragen stammen aus der Wissensbasis selbst. Der Test zeigt also, dass Abruf und Generierung durchgängig zusammenspielen, sagt aber wenig darüber, wie der Bot mit neuen Formulierungen echter Nutzer umgeht. Außerdem ist ein Bewertungsmodell kein Mensch. Der geplante Pilotbetrieb mit Mitgliedern der Community wird zeigen, wie der Bot bei echten Fragen abschneidet.

## Tests und CI/CD

```bash
pip install pytest
pytest tests/
```

| Workflow | Wann | Was er tut |
| --- | --- | --- |
| `ci.yml` | bei jedem Push und Pull Request auf `main` | kompiliert den Code, führt `ruff` aus, startet die Unit-Tests, validiert die JSON-Dateien der Wissensbasis (übersprungen, wenn sie fehlt) und durchsucht die Git-Historie mit `gitleaks` nach Secrets |
| `eval.yml` | manuell und jeden Montag | baut den Index und führt die Evaluation aus. Sie verbraucht OpenAI-Guthaben und ist deshalb von `ci.yml` getrennt. Sie braucht die private Wissensbasis und überspringt sich ohne sie |
| `cd.yml` | manuell, nach Eingabe von `deploy` | verbindet sich per SSH mit dem Server und startet `scripts/deploy.sh` |

`deploy.sh` läuft auf dem Server: Es aktiviert die virtuelle Umgebung, installiert die Abhängigkeiten, startet den `systemd`-Dienst neu und gibt den Dienststatus aus. Der Workflow braucht die Repository-Secrets `VPS_HOST`, `VPS_USER` und `VPS_SSH_KEY` und funktioniert deshalb nur in einem Repository, das sie enthält.

Der Bot läuft als `systemd`-Dienst auf einem VPS. Die Server-Einrichtung (SSH nur mit Schlüssel, Firewall, fail2ban, automatische Sicherheitsupdates) erfolgt auf dem Server und ist nicht Teil dieses Repositorys.

Die Pre-Commit-Hooks (`pip install pre-commit && pre-commit install`) führen lokal vor jedem Commit dieselben `ruff`-Regeln und einige Dateiprüfungen aus.

## Einschränkungen

- **Testphase.** Der Bot wurde noch nicht von echten Nutzern verwendet, seine Qualität bei echten Fragen ist daher unbekannt.
- **Statisches Wissen.** Die Wissensbasis aktualisiert sich nicht selbst. Neue Chatnachrichten müssen die Pipeline erneut durchlaufen.
- **Veraltete Antworten.** Manche Informationen aus den Chats veralten. Zeitgebundene Einträge werden markiert, aber die Markierung ist nur so gut wie die Extraktion.
- **Grobe Kategorien.** Die Kategorien entstehen aus Schlagwortregeln auf dem Themenfeld. Ein Thema mit „Anmeldung“ landet zum Beispiel unter Wohnen, auch wenn es um einen Sprachkurs geht.
- **Blockierender OpenAI-Aufruf.** Der Bot ruft OpenAI synchron auf, das blockiert seine Event-Loop während des Wartens. Für gelegentliche Fragen in einer kleinen Gruppe ist das in Ordnung, bei mehr Last sollte auf den asynchronen Client gewechselt werden.
- **Für Linux festgelegt.** `requirements.txt` ist ein vollständiger Freeze vom Linux-Server mit CPU-only PyTorch. Auf anderen Systemen müssen einzelne Versionen eventuell angepasst werden.
- **Modellabhängigkeit.** Extraktion, Antworten und Bewertung nutzen `gpt-4o-mini`. Mit anderen Modellen können die Ergebnisse abweichen.

## Datenschutz

Die Wissensbasis entsteht aus echten Gesprächen einer privaten Community, und der Rohexport enthält Absendernamen. Exporte, die JSONL-Zwischendateien, die Wissensbasis und den Suchindex nicht in Git aufnehmen und keine `.env`-Dateien committen. Die `.gitignore` in diesem Repository schließt sie aus. Die Beispieldaten enthalten ausschließlich erfundene Inhalte.
