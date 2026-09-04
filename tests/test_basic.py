"""
Basic smoke tests to ensure module importability.
"""

def test_imports():
    """Verify that project modules can be imported without errors."""
    import src
    assert src.__version__ == "0.1.0"
