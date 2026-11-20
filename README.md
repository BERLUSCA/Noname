# Quanto ne sai di film e serie TV?

Un agente genera quiz a partire da pagine Wikipedia recuperate con RAG, verifica le domande con un giudice LLM,
calcola i punteggi nel codice, gestisce sfide a due giocatori e una chat con memoria di sessione.

Progetto del corso AITHO GenAI Academy (Blocco 3), con [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai) v0.1.0.

## Requisiti

- Python 3.11 o superiore (testato con 3.13)
- Connessione internet per la prima installazione (pacchetti, modello di embedding, pagine Wikipedia)
- Una chiave API del provider LLM scelto (Anthropic o OpenAI)

## 1. Installazione

Dalla cartella del progetto:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Su macOS/Linux: `source .venv/bin/activate`.

Per i comandi qui sotto usa sempre il Python del progetto: `.venv\Scripts\python.exe` su Windows,
`.venv/bin/python` su macOS/Linux. Evita il `python` di sistema, che non ha le dipendenze.

## 2. Configurazione

```
copy .env.example .env
```

Apri `.env` e compila:

| Variabile | Cosa mettere |
|---|---|
| `LLM_PROVIDER` | `anthropic` o `openai` |
| `LLM_API_KEY` | la chiave del provider scelto |
| `MODEL_AGENT` | il modello che genera quiz e risponde in chat |
| `MODEL_JUDGE` | il modello che verifica le domande (può essere diverso) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` (embedding locale, nessun costo) |
| `BUDGET_EUR` | budget di sessione in euro (es. `10`) |
| `USD_PER_EUR` | tasso di cambio USD per 1 EUR, con punto o virgola (es. `0.91`) |

Senza `BUDGET_EUR` e `USD_PER_EUR` il sistema blocca le chiamate ai modelli: è voluto (guardrail 7).
Il tasso di cambio va verificato e datato da te.

Non committare mai il file `.env`: è escluso da git.

## 3. Dati: corpus e indice

Il corpus è un sottoinsieme fisso di 19 pagine di Wikipedia in italiano (testo con licenza CC BY-SA),
elencate in `src/ingestion/fetch_wikipedia.py`.

```
python -m src.ingestion.fetch_wikipedia
python -m src.rag.build_index --model paraphrase-multilingual-MiniLM-L12-v2
```

Il primo comando scarica le pagine in `data/raw/` con URL e data nel `manifest.json`.
Il secondo crea l'indice vettoriale in `data/index/` (la prima volta scarica il modello di embedding).
Entrambi sono già presenti nel repository, quindi possono essere saltati.

## 4. Test (offline)

```
python -m pytest -q
```

I test non chiamano nessun modello e non usano la rete: usano client finti.
Per ripeterli a ogni modifica (si riesegue da solo quando salvi un file): `.venv\Scripts\ptw.exe`.

## 5. Avvio del server

```
python -m uvicorn src.api.main:app --reload
```

Oppure, su Windows, doppio clic su `start_api.bat` (scrive il log in `logs\server.txt`).

Poi apri **http://127.0.0.1:8000/docs** per provare gli endpoint dall'interfaccia.

Esempio di flusso:

1. `POST /quiz` con `{"title": "Inception", "n_questions": 2}`: restituisce `quiz_id` e domande, **senza** le risposte corrette.
2. `POST /answer` con `quiz_id`, `player` e `answers` (indici 0-3, `null` per non rispondere): restituisce punteggio, esiti e fonti.
3. `POST /challenge` crea una sfida con un codice; l'avversario risponde con `POST /challenge/{codice}/answer`.
4. `POST /chat` con `{"message": "..."}`: un agente che sceglie se cercare nel corpus, creare un quiz o rispondere.
   Riusa il `session_id` restituito per continuare la conversazione. `DELETE /chat/{session_id}` cancella la storia.
5. `GET /leaderboard`: classifica dei giocatori.

Ogni chiamata a un modello viene registrata in `logs/llm_calls.jsonl` (token, costo stimato, latenza, modello).

## 6. Valutazioni (chiamate reali, a pagamento)

Gli script in `evals/` non fanno parte dei test: chiamano il modello e hanno un costo.

| Script | Cosa misura | Costo |
|---|---|---|
| `evals/rag_eval.py --model <embedding>` | recupero dal corpus sul gold set | gratis (solo embedding locale) |
| `evals/run_eval.py` | generazione quiz: schema, verifica del giudice, accordo con la risposta attesa | 1 generazione + 1 giudizio per voce |
| `evals/run_verifier_eval.py --model <embedding>` | accuratezza del giudice su fatti veri e falsi | 1 chiamata per fatto |
| `evals/run_injection_eval.py` | prompt injection sui 10 casi di `evals/injection_cases.jsonl` | 1 generazione per caso |
| `evals/quality_excerpt.py --embedding-model <name>` | qualità delle risposte della chat con estratto corto e lungo | 1 chat + 1 giudizio per domanda |
| `evals/measure_memory.py --embedding-model <name>` | token in input con e senza finestra di memoria | 10 turni di chat per modalità |
| `evals/smoke_api.py` / `evals/smoke_chat.py` | flusso completo di API e chat con modelli reali | pochissime chiamate |

Per confrontare modelli diversi: `--model <nome>` in `run_eval.py` (agente) e `--judge <nome>` in `quality_excerpt.py` e `run_verifier_eval.py` (giudice). `run_eval.py` usa `MODEL_JUDGE` dal `.env`.

## Struttura

```
src/
  api/        FastAPI (main.py), database SQLite (db.py)
  agent/      client LLM (llm.py), agente e tool (chat.py, tools.py), memoria (memory.py)
  rag/        indice, retrieval, embedding locale, verificatore (verifier.py)
  ingestion/  scaricamento pagine, chunking
  quiz/       generazione (generator.py), verifica dei quiz (verified.py), punteggio (scoring.py),
              giudice delle risposte (answer_judge.py)
  observability/  log JSONL (logger.py), prezzi (pricing.py), budget (budget.py)
data/         corpus (raw/), indice (index/), database (app.db, ignorato da git)
evals/        gold set, casi di injection, script di valutazione
tests/        test offline
```

## Guardrail

1. Il punteggio delle domande chiuse lo calcola il codice, mai il modello.
2. La risposta corretta non va al giocatore prima che risponda, e non finisce nel contesto dell'avversario.
3. Ogni domanda ha una fonte (URL Wikipedia). Se il titolo non è nel corpus, il sistema lo dice e non inventa.
   Le risposte della chat senza documento (conversazione, giocatori, sfide) vengono verificate dal giudice contro i dati dei tool; se non sono supportate, la risposta del modello viene mostrata con la nota «Risposta non verificata».
4. Testo utente, storia e passaggi recuperati sono dati, non istruzioni. Ci sono test per le injection.
5. Il reset di punteggi richiede conferma esplicita (non è esposto nell'API).
6. Limiti: fino a 5 passi dell'agente e 5 tool per richiesta; 3 generazioni e 3 giudizi per quiz; `max_tokens` basso; `top_k` tra 3 e 5.
7. Budget di sessione in euro, con blocco fail-closed.

## Limiti noti

- Il controllo lessicale del retrieval perde casi buoni: il giudice LLM è il controllo principale.
- Il controllo sulle risposte di conversazione dipende dal giudice LLM: può rifiutare parafrasi corrette o accettare affermazioni solo in parte supportate.
- Il giudice e l'agente possono usare lo stesso modello: per una valutazione seria, usare `--judge` con un modello diverso.
- Il budget vale per il processo del server: si azzera al riavvio.
- Il corpus contiene solo 19 opere: per gli altri titoli la chat risponde con la nota «Risposta non verificata», e i quiz non vengono creati.
- Il costo stimato usa la tabella prezzi in `src/observability/pricing.py`, verificata il 2026-11-06: va riverificata prima dell'uso.
