"""
src/pitstop/__init__.py
======================
Pitstop duration estimation package for F1-SIS.
Provides data-driven pit stop duration modeling and Monte Carlo sampling.
"""

from .adapter import PitstopAdapter

__all__ = ["PitstopAdapter"]
