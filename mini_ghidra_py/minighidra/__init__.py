"""MiniGhidraPy: an educational x86/x86-64 static analyzer."""

from .abi import CallingConvention, calling_convention_for, default_calling_convention
from .analysis import Analyzer
from .decompiler import SemanticDecompiler
from .function_starts import FunctionStartCandidate
from .image import BinaryImage, LoadError
from .model import Function, FunctionParameter, FunctionSignature
from .noreturn import NoReturnCatalog
from .project import ProjectState

__all__ = [
    "Analyzer",
    "BinaryImage",
    "CallingConvention",
    "Function",
    "FunctionParameter",
    "FunctionSignature",
    "FunctionStartCandidate",
    "LoadError",
    "NoReturnCatalog",
    "ProjectState",
    "SemanticDecompiler",
    "calling_convention_for",
    "default_calling_convention",
]
__version__ = "0.1.0"
