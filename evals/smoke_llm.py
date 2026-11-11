import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.llm import create_client  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402

PASSAGE = (
    "Inception è un film di fantascienza del 2010 scritto e diretto da Christopher Nolan. "
    "Il protagonista, Dom Cobb, ruba segreti dalla mente delle persone mentre sognano. "
    "Il film è ambientato in più livelli di sogno e ha vinto l'Oscar per i migliori effetti visivi."
)


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = create_client("MODEL_AGENT")
    questions = QuizGenerator(client).generate("Inception", PASSAGE, source="smoke:Inception", n_questions=2)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q.question}")
        for j, option in enumerate(q.options):
            print(f"   {j}) {option}")
        print(f"   corretta: {q.correct_index}, fonte: {q.source}")


if __name__ == "__main__":
    main()
