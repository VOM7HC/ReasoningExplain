from .core import Explainer, __version__


def main() -> None:
    ex = Explainer()
    print(f"reasoning-explain {__version__}")
    print(ex.explain("demo"))


if __name__ == "__main__":
    main()

