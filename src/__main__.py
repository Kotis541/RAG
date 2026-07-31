import fire
from .firemodels import RagPipeline
from .parser import RagParser
from .chunker import RagChunker
from .index import tokenize


def main() -> None:
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