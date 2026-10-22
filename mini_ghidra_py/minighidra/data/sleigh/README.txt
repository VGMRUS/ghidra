Compiled SLEIGH language data for MiniGhidraPy

Source checkout: Ghidra 12.3 DEV
Source language definitions:
  Ghidra/Processors/x86/data/languages/x86.slaspec
  Ghidra/Processors/x86/data/languages/x86-64.slaspec
Language-definition version in x86.ldefs: 4.9

The .sla files were compiled from the checkout's x86 language sources with
Ghidra/Features/Decompiler/src/decompile/cpp/sleigh_opt. The corresponding
processor context specifications (x86.pspec and x86-64.pspec) are copied from
the same source directory. They are loaded by pypcode's native SLEIGH runtime;
the compiled language files preserve the checkout's instruction semantics.

Regenerate after changing Ghidra's x86 SLEIGH sources by building the local
SLEIGH compiler and running, from Ghidra/Processors/x86/data/languages:

  <path-to-sleigh_opt> x86.slaspec <path-to-minighidra>/data/sleigh/x86.sla
  <path-to-sleigh_opt> x86-64.slaspec <path-to-minighidra>/data/sleigh/x86-64.sla

The .sla files are generated processor definitions, not handwritten code.
