# Quanto ne sai di film e serie TV?

Un agente genera quiz a partire da pagine Wikipedia recuperate con RAG, verifica le domande con un giudice LLM,
calcola i punteggi nel codice, gestisce sfide a due giocatori e una chat con memoria di sessione.
L'interfaccia web è composta da due pagine: 
- La prima dove si accede/registra con il nome
- La seconda dove si ha la chat con l'agente, elenco dei film e classifica.

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
| `DB_PATH` | facoltativa: percorso del database SQLite (default `data/app.db`) |

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

Le pagine in `data/raw/` sono già nel repository e il primo comando si può saltare. L'indice invece **non** è nel
repository (`data/index/` è ignorato da git): il secondo comando va eseguito prima di avviare il server.
Se cambi i titoli del corpus o il modello di embedding, ricostruisci l'indice: il server controlla l'impronta
dell'indice e segnala «Modelli non pronti» se non corrisponde.

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

All'avvio il server carica indice, embedding e client dei modelli (può richiedere qualche secondo): la pagina mostra
«Modelli: pronti» quando ha finito.

- **http://127.0.0.1:8000/** : l'applicazione.
- **http://127.0.0.1:8000/docs** : documentazione interattiva dell'API.

## 6. Come si usa

La pagina ha tre aree.

- **Accesso**: si scrive il nome (senza password) e si preme **Entra**. Il nome diventa attivo solo a quel punto.
  **Esci** lo azzera e cancella la sessione di chat.
- **Chat con l'agente**: l'agente sceglie da solo lo strumento da usare.
  Quando crea un quiz, le domande compaiono nella conversazione con le risposte selezionabili; dopo l'invio il bot dice
  quali sono giuste e sbagliate. Sotto il quiz si può **sfidare** un altro giocatore registrato. Le sfide ricevute
  compaiono all'apertura della chat con il pulsante **Accetta**; si può anche scrivere «accetto la sfida».
  Per chiedere le proprie sfide: «le mie sfide». Per l'elenco dei titoli: «che film ci sono?».
- **Film e serie** (colonna di destra): i titoli del corpus, con ricerca. **Quiz** e **Info** scrivono la richiesta in chat.
  La **classifica** è sotto l'elenco e si aggiorna dopo ogni partita.

Strumenti dell'agente (`src/agent/tools.py`): `search_corpus`, `create_quiz`, `ask_which_title`, `list_titles`,
`list_players`, `my_challenges`, `accept_challenge`, `challenge_help`, `recall_conversation`.

Regole del gioco:

- Ogni quiz ha da 1 a 10 domande a 4 opzioni (in chat fino a 5), una sola corretta. Risposta corretta = 1 punto,
  sbagliata o senza risposta = 0.
- Un giocatore risponde una sola volta a ciascun quiz.
- Una sfida usa lo stesso quiz: lo sfidante gioca per primo e invita un avversario registrato, che risponde dopo.
  Finché entrambi non hanno risposto non si vede nessun punteggio; poi si vedono solo i totali e il vincitore.
- I punti confluiscono nella classifica.

### Esempio con l'API

1. `POST /players` con `{"name": "Anna"}`: registra il giocatore. `GET /players` li elenca, `GET /titles` elenca i titoli.
2. `POST /quiz` con `{"title": "Inception", "n_questions": 2}`: restituisce `quiz_id` e domande, **senza** le risposte corrette.
3. `POST /answer` con `quiz_id`, `player` e `answers` (indici 0-3, `null` per non rispondere): restituisce punteggio, esiti e fonti.
4. Sfide: `POST /quiz/{quiz_id}/challenge` (`challenger`, `opponent`) sfida un giocatore con un quiz già giocato;
   `POST /challenges` risponde e sfida in un colpo solo; `POST /challenge` crea una sfida aperta con un codice.
   L'avversario risponde con `POST /challenge/{codice}/answer`; `GET /challenge/{codice}` mostra lo stato;
   `GET /players/{nome}/challenges` (intestazione `X-Player` uguale al nome) elenca le sfide di un giocatore.
5. `POST /chat` con `{"player": "Anna", "message": "..."}`: l'agente cerca nel corpus, crea quiz, gestisce le sfide o risponde.
   Riusa il `session_id` restituito per continuare la conversazione; la risposta contiene anche le domande dei quiz creati.
   `DELETE /chat/{session_id}` cancella la storia (le sessioni si cancellano comunque dopo 24 ore).
6. `GET /leaderboard`: classifica dei giocatori.

Ogni chiamata a un modello viene registrata in `logs/llm_calls.jsonl` (token, costo stimato, latenza, modello).

## 7. Valutazioni (chiamate reali, a pagamento)

Gli script in `evals/` non fanno parte dei test: chiamano il modello e hanno un costo.

| Script | Cosa misura | Costo |
|---|---|---|
| `evals/rag_eval.py --model <embedding>` | recupero dal corpus sul gold set | gratis (solo embedding locale) |
| `evals/run_eval.py` | generazione quiz: schema, verifica del giudice, accordo con la risposta attesa | 1 generazione + 1 giudizio per voce |
| `evals/run_verifier_eval.py --model <embedding>` | accuratezza del giudice su fatti veri e falsi | 1 chiamata per fatto |
| `evals/run_injection_eval.py` | prompt injection sui 10 casi di `evals/injection_cases.jsonl` | 1 generazione per caso |
| `evals/quality_excerpt.py --embedding-model <name>` | qualità delle risposte della chat con estratto corto e lungo | 1 chat + 1 giudizio per domanda |
| `evals/measure_memory.py --embedding-model <name>` | token in input con e senza finestra di memoria | 10 turni di chat per modalità |
| `evals/measure_repetition.py --embedding-model <name>` | quante domande si ripetono tra quiz successivi sullo stesso titolo | `--quizzes` generazioni per modalità, 2 modalità (default 10) |
| `evals/smoke_llm.py` | una generazione di prova con il modello dell'agente | 1 chiamata |
| `evals/smoke_api.py` / `evals/smoke_chat.py` | flusso completo di API e chat con modelli reali | pochissime chiamate |

Per confrontare modelli diversi: `--model <nome>` in `run_eval.py` (agente) e `--judge <nome>` in `quality_excerpt.py` e `run_verifier_eval.py` (giudice). `run_eval.py` usa `MODEL_JUDGE` dal `.env`.

## Struttura

```
src/
  api/        FastAPI (main.py), database SQLite (db.py)
    static/   la pagina web: index.html, common.js, style.css (HTML e JavaScript semplici, nessuna build)
  agent/      client LLM (llm.py), agente e tool (chat.py, tools.py), memoria (memory.py)
  rag/        indice, retrieval, embedding locale, verificatore (verifier.py)
  ingestion/  scaricamento pagine, chunking
  quiz/       generazione (generator.py), controlli (checks.py), verifica dei quiz (verified.py),
              punteggio (scoring.py), giudice delle risposte per le valutazioni (answer_judge.py)
  observability/  log JSONL (logger.py), prezzi (pricing.py), budget (budget.py)
data/         corpus (raw/), indice (index/), database (app.db, ignorato da git)
docs/         presentazione (.pptx) e ripasso delle scelte di progetto (Ripasso.md)
evals/        gold set (gold.jsonl; gold_proposed.jsonl contiene proposte da verificare), casi di injection, script di valutazione
tests/        test offline
```

## Guardrail

1. Il punteggio delle domande chiuse lo calcola il codice, mai il modello.
2. La risposta corretta non va al giocatore prima che risponda, e non finisce nel contesto dell'avversario.
3. Ogni domanda ha una fonte (URL Wikipedia). Se il titolo non è nel corpus, il sistema lo dice e non inventa.
   Le risposte della chat senza documento (conversazione, giocatori, sfide) vengono verificate dal giudice contro i dati dei tool; se non sono supportate, la risposta del modello viene mostrata con la nota «Risposta non verificata».
4. Testo utente, storia e passaggi recuperati sono dati, non istruzioni. Ci sono test per le injection.
5. Il reset dei punteggi richiede conferma esplicita: nell'API non esiste nessuna operazione di reset.
6. Limiti: fino a 5 passi dell'agente e 5 tool per richiesta; 3 generazioni e 3 giudizi per quiz; `max_tokens` basso; `top_k` tra 3 e 5.
7. Budget di sessione in euro, con blocco fail-closed.

## Limiti noti

- Il controllo lessicale del retrieval perde casi buoni: il giudice LLM è il controllo principale.
- Il controllo sulle risposte di conversazione dipende dal giudice LLM: può rifiutare parafrasi corrette o accettare affermazioni solo in parte supportate.
- Il giudice e l'agente possono usare lo stesso modello: per una valutazione seria, usare `--judge` con un modello diverso.
- Il budget vale per il processo del server: si azzera al riavvio.
- La memoria della chat tiene solo gli ultimi 6 messaggi (3 scambi) più i quiz creati nella sessione.
- I nomi dei giocatori non hanno password: chi conosce un nome può usarlo.
- Le sfide ricevute compaiono all'apertura della chat o con «le mie sfide»: la pagina non si aggiorna da sola.
- Non si può rifiutare una sfida.
- Il gold set ha 10 voci e non copre tutti i titoli del corpus.
- Il corpus contiene solo 19 opere: per gli altri titoli la chat risponde con la nota «Risposta non verificata», e i quiz non vengono creati.
- Il costo stimato usa la tabella prezzi in `src/observability/pricing.py`, verificata il 2026-11-06: va riverificata prima dell'uso.
