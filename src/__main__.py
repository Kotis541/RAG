import fire
from .pipeline import RagPipeline


def main() -> None:
    """Entry point for the RAG CLI."""
    fire.Fire(RagPipeline)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt - bye :D")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"[FATAL ERROR]: {e} ")