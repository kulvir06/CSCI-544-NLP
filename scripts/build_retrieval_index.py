"""Build the local passage index from approved document reviews."""

from retrieval.index import build_index


def main() -> None:
    print(build_index().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
