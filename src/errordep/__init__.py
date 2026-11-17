"""errordep: are your LLM judges and verifiers really independent?"""
from .core import Interval, MajorityStat, PairStat, Panel, Report

__version__ = "0.1.0"
__all__ = ["Panel", "PairStat", "MajorityStat", "Interval", "Report", "__version__"]
