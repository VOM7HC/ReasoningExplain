from reasoning_explain.core import Explainer


def test_explainer_returns_string():
    ex = Explainer()
    out = ex.explain("hello")
    assert isinstance(out, str)
    assert "hello" in out


def test_explainer_rejects_empty():
    ex = Explainer()
    try:
        ex.explain("")
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for empty input"

