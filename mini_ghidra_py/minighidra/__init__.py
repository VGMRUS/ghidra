"""MiniGhidraPy: an educational x86/x86-64 static analyzer."""

from .abi import (
    CallingConvention,
    calling_convention_by_name,
    calling_convention_for,
    default_calling_convention,
)
from .analysis import Analyzer
from .decompiler import SemanticDecompiler
from .function_starts import FunctionStartCandidate
from .image import BinaryImage, LoadError
from .model import Function, FunctionParameter, FunctionSignature
from .noreturn import NoReturnCatalog
from .paramid import (
    InstructionDataFlow,
    ParamActive,
    ParamIDAnalysis,
    ParamRank,
    ParamTrial,
    PcodeLikeGraph,
)
from .project import ProjectState
from .prototype import PrototypeLayout, PrototypeResolver
from .sleigh import SleighInstruction, SleighPcodeOp, SleighTranslator, SleighVarnode
from .typesys import CField, CType, StorageLocation, TypeDatabase, TypeKind

__all__ = [
    "Analyzer",
    "BinaryImage",
    "CField",
    "CType",
    "CallingConvention",
    "Function",
    "FunctionParameter",
    "FunctionSignature",
    "FunctionStartCandidate",
    "InstructionDataFlow",
    "LoadError",
    "NoReturnCatalog",
    "ParamActive",
    "ParamIDAnalysis",
    "ParamRank",
    "ParamTrial",
    "PcodeLikeGraph",
    "ProjectState",
    "PrototypeLayout",
    "PrototypeResolver",
    "SemanticDecompiler",
    "SleighInstruction",
    "SleighPcodeOp",
    "SleighTranslator",
    "SleighVarnode",
    "StorageLocation",
    "TypeDatabase",
    "TypeKind",
    "calling_convention_by_name",
    "calling_convention_for",
    "default_calling_convention",
]
__version__ = "0.1.0"
