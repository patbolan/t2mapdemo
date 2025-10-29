"""
Flask application for prostate T2 relaxometry visualization.
This package provides web-based tools for comparing parametric maps.
"""

# Import app only when accessed as a package, not when running directly
try:
    from .app import app
except ImportError:
    # Fallback for when running from same directory
    from app import app

__version__ = '1.0.0'
__all__ = ['app']
