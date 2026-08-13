"""Package import smoke test."""


def test_package_imports() -> None:
    import galaxy_classifier

    assert galaxy_classifier.__name__ == "galaxy_classifier"
