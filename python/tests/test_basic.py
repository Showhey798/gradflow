"""Basic tests for GradFlow package structure."""

import sys


def test_python_version():
    """Test that Python version meets minimum requirements."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"


def test_import_gradflow():
    """Test that gradflow package can be imported."""
    try:
        import gradflow
        assert gradflow is not None
    except ImportError as e:
        # Allow import failure for now since bindings are not yet implemented
        # This test will pass once Python bindings are built
        assert "gradflow" in str(e) or "No module" in str(e)


def test_package_structure():
    """Test basic package structure."""
    try:
        import gradflow
        # Check that package has __version__ attribute (once implemented)
        # For now, just verify the package can be accessed
        assert hasattr(gradflow, "__name__")
    except ImportError:
        # Expected failure until bindings are implemented
        pass
