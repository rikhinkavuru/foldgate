"""Smoke test: the package and its subpackages import cleanly (torch-free)."""


def test_package_imports():
    import foldgate

    assert foldgate.__version__
    for sub in foldgate.__all__:
        __import__(f"foldgate.{sub}")
