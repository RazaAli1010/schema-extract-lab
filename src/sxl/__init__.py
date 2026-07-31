"""schema-extract-lab — strict JSON schema extraction from unstructured documents.

Deliberately imports nothing. Submodules are imported explicitly by callers so
that `import sxl` stays free of side effects and torch never enters `sys.modules`
on the development laptop (SPEC §2.1, §9.2).
"""

__version__ = "0.1.0"
