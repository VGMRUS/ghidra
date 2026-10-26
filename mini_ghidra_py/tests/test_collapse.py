"""Regresiones del motor de estructuración de CFG (``block``/``tracedag``/``collapse``).

El puerto de ``BlockGraph``/``CollapseStructure``/``TraceDAG`` no tenía ninguna
cobertura: los cuelgues que motivaron el puerto fiel (179 de 300 CFG
aleatorios) no los detectaba ningún test.  Aquí se fijan

* los invariantes de ``build_copy`` (orden de salidas, aristas paralelas,
  ``f_switch_out``),
* la semántica de los constructores de bloques (``swapEdges``,
  ``newBlockCondition``, ``getFrontLeaf``),
* los CFG que colgaban el estructurador antiguo, y
* los CFG patológicos que ahora fallan rápido con ``StructureError`` en vez de
  girar para siempre.

Todas las pruebas se ejecutan bajo un vigilante (``SIGALRM``) para que una
regresión se reporte como fallo en lugar de bloquear la suite.
"""
from __future__ import annotations

import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

from minighidra.analysis import Analyzer
from minighidra.block import (
    F_FLIP_PATH,
    F_GOTO_EDGE,
    F_SWITCH_OUT,
    F_TREE_EDGE,
    BlockBasic,
    BlockGraph,
    StructureError,
)
from minighidra.collapse import CollapseStructure
from minighidra.image import BinaryImage
from minighidra.tracedag import push_budget

# CFG que colgaban el estructurador anterior (fuzzer, semilla 1).  El primero
# es el caso mínimo: n=4, aristas 0->1, 0->2, 1->2, 2->3, 3->0.
HANGING_SPECS = [
    {"n": 4, "edges": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 0]]},
    {"n": 4, "edges": [[0, 1], [0, 2], [1, 2], [1, 3], [1, 0]]},
    {"n": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]},
    {"n": 5, "edges": [[0, 4], [0, 2], [1, 4], [1, 3], [2, 3], [2, 4], [4, 0]]},
    {
        "n": 6,
        "edges": [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [3, 5],
                  [4, 5], [5, 0]],
    },
    {
        "n": 6,
        "edges": [[0, 1], [0, 4], [1, 2], [1, 0], [2, 3], [2, 4], [2, 5],
                  [3, 4], [4, 5], [4, 0]],
    },
    {
        "n": 7,
        "edges": [[0, 1], [1, 2], [1, 4], [1, 3], [2, 6], [3, 1], [4, 5],
                  [4, 6], [5, 6], [6, 0]],
    },
    {
        "n": 7,
        "edges": [[0, 1], [0, 3], [1, 2], [1, 6], [1, 4], [2, 3], [3, 4],
                  [3, 6], [4, 6], [5, 6], [6, 3]],
    },
    {
        "n": 8,
        "edges": [[0, 1], [0, 2], [1, 2], [1, 4], [2, 3], [2, 1], [3, 4],
                  [4, 7], [4, 6], [5, 6], [6, 7]],
    },
    {
        "n": 8,
        "edges": [[0, 1], [1, 2], [1, 6], [1, 0], [2, 3], [2, 1], [3, 4],
                  [3, 3], [4, 5], [4, 7], [4, 4], [5, 6], [6, 7]],
    },
    {
        "n": 9,
        "edges": [[0, 1], [1, 2], [1, 7], [2, 3], [2, 7], [2, 6], [2, 0],
                  [3, 8], [4, 5], [4, 7], [4, 3], [5, 6], [5, 4], [6, 8],
                  [7, 8]],
    },
    {
        "n": 9,
        "edges": [[0, 1], [0, 5], [1, 2], [1, 5], [1, 5], [2, 3], [3, 4],
                  [4, 5], [4, 1], [5, 6], [5, 7], [5, 8], [6, 7], [6, 8],
                  [7, 8], [8, 7]],
    },
]

# CFG irreducibles en los que el trazo rebota entre dos bloques: Ghidra tampoco
# los estructura (confía en que el subgrafo "loop DAG" es acíclico).  El puerto
# acota las iteraciones y lanza StructureError en lugar de colgar.
PATHOLOGICAL_SPECS = [
    {
        "n": 9,
        "edges": [[0, 1], [0, 0], [1, 2], [1, 3], [1, 7], [2, 3], [2, 4],
                  [3, 5], [3, 8], [3, 0], [4, 5], [4, 8], [5, 6], [5, 8],
                  [5, 3], [6, 7], [6, 8], [6, 1], [7, 8], [7, 5]],
    },
    {
        "n": 16,
        "edges": [[0, 1], [0, 15], [1, 2], [1, 13], [1, 3], [2, 3], [2, 9],
                  [2, 7], [3, 4], [3, 9], [3, 11], [4, 5], [4, 10], [4, 0],
                  [5, 6], [5, 10], [5, 12], [6, 7], [6, 15], [6, 12], [7, 8],
                  [7, 6], [8, 12], [8, 10], [9, 11], [10, 11], [10, 12],
                  [11, 12], [11, 14], [12, 13], [12, 14], [12, 15], [12, 10],
                  [13, 14], [13, 15], [13, 7], [14, 15], [15, 8]],
    },
]


@contextmanager
def watchdog(seconds: float = 5.0) -> Iterator[None]:
    """Falla la prueba si el bloque protegido tarda más de ``seconds``."""
    if not hasattr(signal, "setitimer"):  # pragma: no cover - solo Windows
        yield
        return

    def _expired(signum: int, frame) -> None:  # noqa: ANN001
        raise AssertionError(f"el estructurador tardó más de {seconds} s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _instruction(
    conditional: bool = False,
    target: Optional[int] = None,
    next_address: Optional[int] = None,
) -> SimpleNamespace:
    """Instrucción mínima con los campos que consulta ``build_copy``."""
    return SimpleNamespace(
        is_conditional_jump=conditional,
        branch_target=target,
        next_address=next_address,
        computed_targets=(),
    )


def _raw(start: int, successors, last=None) -> SimpleNamespace:
    """Bloque plano con la interfaz que consume ``BlockGraph.build_copy``."""
    instructions = () if last is None else (last,)
    return SimpleNamespace(
        start=start, successors=tuple(successors), instructions=instructions
    )


def _by_address(graph: BlockGraph, address: int):
    """Bloque ``basic`` del grafo con la dirección dada."""
    for block in graph.blocks:
        if getattr(block, "start_addr", None) == address:
            return block
    raise AssertionError(f"sin bloque en {address:#x}")


def _graph_from_spec(spec: dict) -> BlockGraph:
    """Construye el grafo por la vía de producción (``build_copy``)."""
    base = 0x1000
    addresses = [base + 0x10 * index for index in range(spec["n"])]
    outgoing: dict[int, list[int]] = {index: [] for index in range(spec["n"])}
    for source, target in spec["edges"]:
        if target not in outgoing[source]:
            outgoing[source].append(target)
    raws = [
        _raw(addresses[index], [addresses[succ] for succ in outgoing[index]])
        for index in range(spec["n"])
    ]
    graph = BlockGraph()
    graph.build_copy(raws)
    return graph


def _collapse(graph: BlockGraph) -> None:
    with watchdog():
        CollapseStructure(graph).collapse_all()


def _shapes(block) -> list[str]:
    """Tipos de bloque del árbol, en orden de emisión."""
    kinds = [getattr(block, "block_type", "?")]
    for child in getattr(block, "blocks", None) or ():
        kinds.extend(_shapes(child))
    return kinds


def _addresses(block) -> list[int]:
    """Direcciones de las hojas ``basic`` del árbol."""
    start = getattr(block, "start_addr", None)
    if getattr(block, "block_type", "") == "basic" and isinstance(start, int):
        return [start]
    out: list[int] = []
    for child in getattr(block, "blocks", None) or ():
        out.extend(_addresses(child))
    return out


class BuildCopyTests(unittest.TestCase):
    """Invariantes de ``BlockGraph.build_copy`` (block.cc:1943)."""

    def test_false_out_is_the_fallthrough(self) -> None:
        # Analyzer genera los sucesores de un salto condicional como
        # (objetivo, caída); Ghidra exige out[0] = rama falsa.
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(
                    0x100,
                    (0x120, 0x110),
                    _instruction(True, target=0x120, next_address=0x110),
                ),
                _raw(0x110, ()),
                _raw(0x120, ()),
            ]
        )
        head = _by_address(graph, 0x100)
        self.assertEqual(head.get_false_out().start_addr, 0x110)
        self.assertEqual(head.get_true_out().start_addr, 0x120)
        # out[0] es la rama falsa y out[1] la verdadera (block.hh).
        self.assertEqual(head.get_out(0).start_addr, 0x110)
        self.assertEqual(head.get_out(1).start_addr, 0x120)

    def test_parallel_edges_are_deduplicated(self) -> None:
        graph = BlockGraph()
        graph.build_copy([_raw(0x100, (0x110, 0x110)), _raw(0x110, ())])
        self.assertEqual(_by_address(graph, 0x100).size_out(), 1)

    def test_more_than_two_outs_is_a_switch(self) -> None:
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(0x100, (0x110, 0x120, 0x130)),
                _raw(0x110, ()),
                _raw(0x120, ()),
                _raw(0x130, ()),
            ]
        )
        head = _by_address(graph, 0x100)
        self.assertTrue(head.is_switch_out())
        self.assertTrue(head.flags & F_SWITCH_OUT)

    def test_spanning_tree_labels_back_edges(self) -> None:
        graph = _graph_from_spec(
            {"n": 3, "edges": [[0, 1], [1, 2], [2, 0]]}
        )
        back = _by_address(graph, 0x1020)
        self.assertEqual(back.size_out(), 1)
        labels = [
            back.out_labels[index] for index in range(back.size_out())
        ]
        self.assertTrue(any(label & 0x80 for label in labels))  # f_back_edge
        self.assertFalse(back.is_loop_dag_out(0))


class FlowBlockPrimitiveTests(unittest.TestCase):
    """Semántica de los mutadores de aristas y constructores de bloques."""

    def _conditional_graph(self) -> tuple[BlockGraph, BlockBasic]:
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(
                    0x100,
                    (0x120, 0x110),
                    _instruction(True, target=0x120, next_address=0x110),
                ),
                _raw(0x110, (0x130,)),
                _raw(0x120, (0x130,)),
                _raw(0x130, ()),
            ]
        )
        return graph, _by_address(graph, 0x100)

    def test_swap_edges_exchanges_targets_and_keeps_labels(self) -> None:
        graph, head = self._conditional_graph()
        before = [(head.get_out(i), head.out_labels[i]) for i in range(2)]
        flip_before = head.flags & F_FLIP_PATH
        head._swap_edges()
        after = [(head.get_out(i), head.out_labels[i]) for i in range(2)]
        # Los destinos se intercambian junto con su etiqueta (block.cc:222).
        self.assertIs(after[0][0], before[1][0])
        self.assertIs(after[1][0], before[0][0])
        self.assertEqual(after[0][1], before[1][1])
        self.assertEqual(after[1][1], before[0][1])
        self.assertNotEqual(head.flags & F_FLIP_PATH, flip_before)
        self.assertIs(head.get_true_out(), before[0][0])

    def test_new_block_condition_picks_or_when_false_out_is_second(self) -> None:
        graph, head = self._conditional_graph()
        false_out = head.get_false_out()
        true_out = head.get_true_out()
        or_node = graph.new_block_condition(head, false_out)
        self.assertEqual(or_node.op, "||")
        graph2, head2 = self._conditional_graph()
        and_node = graph2.new_block_condition(head2, head2.get_true_out())
        self.assertEqual(and_node.op, "&&")
        self.assertIsNot(or_node, and_node)
        self.assertIsNotNone(true_out)

    def test_front_leaf_stops_at_basic_block(self) -> None:
        graph = BlockGraph()
        graph.build_copy([_raw(0x100, (0x110,)), _raw(0x110, ())])
        container = graph.new_block_list(list(graph.blocks))
        self.assertEqual(container.get_front_leaf().start_addr, 0x100)

    def test_set_goto_branch_marks_the_edge(self) -> None:
        graph, head = self._conditional_graph()
        head.set_goto_branch(1)
        self.assertTrue(head.is_goto_out(1))
        self.assertTrue(head.out_labels[1] & F_GOTO_EDGE)
        self.assertFalse(head.out_labels[0] & F_GOTO_EDGE)


class CollapseStructureTests(unittest.TestCase):
    """Colapsos completos de CFG pequeños y de los que antes colgaban."""

    def test_diamond_becomes_if_else_with_ghidra_polarity(self) -> None:
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(
                    0x100,
                    (0x120, 0x110),
                    _instruction(True, target=0x120, next_address=0x110),
                ),
                _raw(0x110, (0x130,)),
                _raw(0x120, (0x130,)),
                _raw(0x130, ()),
            ]
        )
        _collapse(graph)
        self.assertEqual(len(graph.blocks), 1)
        top = graph.blocks[0]
        self.assertEqual(top.block_type, "list")
        self.assertEqual(top.size_in(), 0)
        self.assertEqual(top.size_out(), 0)
        ifelse = top.blocks[0]
        self.assertEqual(ifelse.block_type, "ifelse")
        self.assertEqual(ifelse.true_block.get_front_leaf().start_addr, 0x120)
        self.assertEqual(ifelse.false_block.get_front_leaf().start_addr, 0x110)
        self.assertEqual(_addresses(top.blocks[1]), [0x130])

    def test_loop_becomes_a_loop_node(self) -> None:
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(
                    0x100,
                    (0x120, 0x110),
                    _instruction(True, target=0x120, next_address=0x110),
                ),
                _raw(0x110, (0x100,)),
                _raw(0x120, ()),
            ]
        )
        _collapse(graph)
        self.assertEqual(len(graph.blocks), 1)
        kinds = _shapes(graph.blocks[0])
        self.assertTrue(
            {"whiledo", "dowhile", "infloop"} & set(kinds),
            f"sin nodo de bucle: {kinds}",
        )

    def test_switch_block_groups_its_cases(self) -> None:
        graph = BlockGraph()
        graph.build_copy(
            [
                _raw(0x100, (0x110, 0x120, 0x130)),
                _raw(0x110, (0x140,)),
                _raw(0x120, (0x140,)),
                _raw(0x130, (0x140,)),
                _raw(0x140, ()),
            ]
        )
        _collapse(graph)
        kinds = _shapes(graph.blocks[0])
        self.assertIn("switch", kinds)

    def test_minimal_hanging_cfg_terminates(self) -> None:
        graph = _graph_from_spec(HANGING_SPECS[0])
        _collapse(graph)
        self.assertEqual(len(graph.blocks), 1)
        for block in graph.blocks:
            self.assertEqual(block.size_in(), 0)
            self.assertEqual(block.size_out(), 0)

    def test_historical_hanging_cfgs_terminate(self) -> None:
        for spec in HANGING_SPECS:
            with self.subTest(n=spec["n"], edges=len(spec["edges"])):
                graph = _graph_from_spec(spec)
                _collapse(graph)
                # collapseAll termina cuando todos los bloques están aislados.
                for block in graph.blocks:
                    self.assertEqual(block.size_in(), 0, f"entrada en {block}")
                    self.assertEqual(block.size_out(), 0, f"salida en {block}")

    def test_pathological_cfgs_fail_fast_with_structure_error(self) -> None:
        for spec in PATHOLOGICAL_SPECS:
            with self.subTest(n=spec["n"]):
                graph = _graph_from_spec(spec)
                with watchdog(5.0):
                    with self.assertRaises(StructureError):
                        CollapseStructure(graph).collapse_all()

    def test_push_budget_grows_with_the_graph(self) -> None:
        small = _graph_from_spec({"n": 4, "edges": [[0, 1], [1, 2], [2, 3]]})
        big = _graph_from_spec(HANGING_SPECS[-1])
        self.assertLess(push_budget(small), push_budget(big))
        self.assertGreaterEqual(push_budget(small), 4096)

    def test_straight_chain_collapses_into_one_list(self) -> None:
        graph = _graph_from_spec({"n": 3, "edges": [[0, 1], [1, 2]]})
        self.assertTrue(graph.blocks[0].out_labels[0] & F_TREE_EDGE)
        _collapse(graph)
        self.assertEqual(len(graph.blocks), 1)
        top = graph.blocks[0]
        self.assertEqual(top.block_type, "list")
        self.assertEqual(_addresses(top), [0x1000, 0x1010, 0x1020])
        self.assertEqual(top.size_in(), 0)
        self.assertEqual(top.size_out(), 0)


class PseudocodePolarityTests(unittest.TestCase):
    """La polaridad impresa coincide con la rama tomada del salto."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def _pseudocode(self, name: str, hexbytes: str, base: int) -> str:
        path = self.root / name
        path.write_bytes(bytes.fromhex(hexbytes))
        analyzer = Analyzer(BinaryImage.load(path, raw_arch="x86_64", base=base))
        return analyzer.pseudocode(analyzer.functions[base])

    def test_taken_branch_body_goes_inside_the_if(self) -> None:
        # cmp edi,0xa ; jg +2 ; jmp +10 ; mov dword [x],42 ; ret
        text = self._pseudocode(
            "taken.bin", "83 ff 0a 7f 02 eb 0a c7 05 00 20 00 00 2a 00 00 00 c3", 0xB000
        )
        lines = [line.strip() for line in text.splitlines()]
        index = next(
            position
            for position, line in enumerate(lines)
            if line.startswith("if (")
        )
        self.assertIn("> (int32_t)0xa", lines[index])
        self.assertIn("= 0x2a;", lines[index + 1])

    def test_loop_condition_is_negated_when_body_is_the_fallthrough(self) -> None:
        # cmp edi,0 ; jle +4 ; dec edi ; jmp -9
        text = self._pseudocode("while.bin", "83 ff 00 7e 04 ff cf eb f7 c3", 0xB100)
        self.assertIn("while ((int32_t)(uint32_t)phi_rdi_loc_B100 > (int32_t)0) {", text)
        self.assertNotIn("goto loc_", text)

    def test_irreducible_loop_terminates(self) -> None:
        # Bucle con dos entradas (B<->C): irreducible.
        with watchdog(10.0):
            text = self._pseudocode("irred.bin", "85 ff 74 02 eb 00 85 ff 75 fa c3", 0xC000)
        self.assertIn("while (", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestInternalConditionEdges(unittest.TestCase):
    """``get_true_out()`` needs two out-edges; callers must guard it.

    A node that the collapse machinery makes *internal* can end up with a
    single out-edge (``copyIn`` plus ``BlockIfGoto`` dropping the goto edge),
    and the pseudocode renderer used to ask such a node for its true
    out-block unconditionally.  Real-world repro: GXSCC.exe functions
    0x401910, 0x4035A0, 0x40CE20, 0x40D3E0 and 0x411F60, all of which raised
    ``IndexError: list index out of range`` from ``block.py:get_true_out``.
    """

    def test_single_out_block_has_no_true_out(self):
        graph = _graph_from_spec({"n": 2, "edges": [(0, 1)]})
        block = _by_address(graph, 0x1000)
        self.assertEqual(block.size_out(), 1)
        with self.assertRaises(IndexError):
            block.get_true_out()
        # The guarded form used by the renderer stays safe.
        self.assertIsNone(
            block.get_true_out() if block.size_out() > 1 else None
        )

    def test_two_out_block_exposes_its_true_out(self):
        graph = _graph_from_spec(
            {"n": 3, "edges": [(0, 1), (0, 2), (1, 2)]}
        )
        block = _by_address(graph, 0x1000)
        self.assertEqual(block.size_out(), 2)
        self.assertIsNotNone(block.get_true_out())
