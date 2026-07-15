import fire
from models import RagPipeline

def main() -> None:
    fire.Fire(RagPipeline)
if __name__ == "__main__":
    main()