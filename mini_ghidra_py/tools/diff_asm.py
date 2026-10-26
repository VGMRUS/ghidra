#!/usr/bin/env python3
"""Diff instrucción a instrucción: listado Ghidrathon de Ghidra vs MiniGhidraPy.

Uso
---
    python tools/diff_asm.py LISTADO_DE_GHIDRA BINARIO [--residual RUTA]

`LISTADO_DE_GHIDRA` es un volcado de Ghidrathon con este aspecto (tres formatos
de línea útiles y varios que no lo son)::

    >>> D(0x401FA0,0x402114);D(0x402170,0x402244);XR(0x40CDC0);...
    ============
     ARCHIVO: Raw Assembly 8Bit ADSR part 2.txt
    ============
    === ASM 00401FA0-00402114 ===
    0x00401FA2: AND ESI,0xffff
    0x0040227A: CALL 0x0040cdc0 -> FUN_00401910
    0x00414C50: -41 -10 -5 4C
    0x0044300C: PTR_RegDeleteValueA_0044300c
    0x00402A04 -> FUN_00402a04

Qué se compara
--------------
1. **Cobertura de direcciones**: cuántas direcciones del listado existen en
   MiniGhidraPy, y qué son las que no.
2. **Texto de la instrucción**: en crudo y tras normalizar el formato. Ghidra y
   Capstone escriben la misma instrucción de formas distintas (mnemotécnicos
   condicionales, prefijos de tamaño opcionales, radices de los inmediatos,
   registros de la FPU, instrucciones de cadena, escalas y desplazamientos);
   `canonizar()` las reduce a una forma común y lo que queda es desacuerdo real.
3. **Fronteras de función**: la anotación `-> FUN_xxxxxxxx` dice qué función
   contiene la dirección de esa línea, así que el listado es un oráculo de
   fronteras, no sólo de desensamblado. Se compara con dos criterios:
   pertenencia al flujo de la función y span `[inicio, fin]` (el de Ghidra y el
   de `Analyzer.function_at`).
4. **Rangos `D(a,b)`**: son los rangos que el autor pidió desensamblar, no
   fronteras de función; se informa de cuántos coinciden con inicio de función o
   de bloque básico para no interpretarlos mal.

Líneas que se descartan (no son instrucciones)
---------------------------------------------
* volcados de bytes del operando: ``0x00414C50: -41 -10 -5 4C``;
* volcados en línea tras ``|``: ``MOV ECX,dword ptr [EBP + 0xc] | -75 4D 0``
  (también aparecen así las líneas que colapsan varias instrucciones);
* etiquetas de datos: ``PTR_``, ``DAT_``, ``s_``, ``LAB_``…;
* líneas ``XR()`` sin dos puntos: ``0x00402A04 -> FUN_00402a04`` (se usan como
  oráculo de función contenedora, no como instrucciones).

Las variantes indentadas ``0x004053B7 (FUN_00405354): …`` y
``0x004141ad (in FUN_004141a3)`` no encajan en ningún formato y se ignoran.

El script no ejecuta el binario: sólo lo desensambla.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minighidra.analysis import Analyzer  # noqa: E402
from minighidra.image import BinaryImage  # noqa: E402

LINEA = re.compile(r"^0x([0-9A-Fa-f]{8}):\s*(.*?)\s*$")
XRLINEA = re.compile(r"^0x([0-9A-Fa-f]{8})\s*->\s*(.+?)\s*$")
# Variantes indentadas que además nombran la función contenedora:
#   `0x00401877 (FUN_00401830): MOV EDX,…`  `0x004012B0 (?): CALL …`
#   `0x00402A32 in ?: CALL …`               `0x00401916 (in FUN_00401910)`
VARIANTE = re.compile(
    r"^0x([0-9A-Fa-f]{8})\s+(?:\((?:in\s+)?(FUN_[0-9a-f]{8}|\?)\)"
    r"|in\s+(FUN_[0-9a-f]{8}|\?))\s*:?\s*(.*)$"
)
RANGO = re.compile(r"===\s*ASM\s*([0-9A-Fa-f]{8})-([0-9A-Fa-f]{8})\s*===")
DIRECTIVA = re.compile(r"\b([DX])R?\((0x[0-9A-Fa-f]+)(?:\s*,\s*(0x[0-9A-Fa-f]+))?\)")
# Volcado de bytes: sólo dígitos, letras hexadecimales, signos y espacios.
VOLCADO = re.compile(r"[0-9A-Fa-f\- ]+")
ETIQUETA_DATO = re.compile(
    r"^(?:PTR_|DAT_|s_|u_|b_|w_|UNK_|LAB_|ORFANO|FUN_|switchD_|caseD_)"
)
ANOTACION = re.compile(r"->\s*(FUN_[0-9a-f]{8}|\?|ORFANO @ 0x[0-9a-f]+)")
DESTINO = re.compile(r"\b0x([0-9A-Fa-f]{6,8})\b")

# Ghidra usa los nombres "condición" del manual de Intel; Capstone los canónicos.
ALIAS = {
    "jz": "je", "jnz": "jne", "jc": "jb", "jnc": "jae", "jnae": "jb",
    "jnb": "jae", "jp": "jp", "jnp": "jnp", "jpe": "jp", "jpo": "jnp",
    "setz": "sete", "setnz": "setne", "setc": "setb", "setnc": "setae",
    "cmovz": "cmove", "cmovnz": "cmovne", "cmovc": "cmovb", "cmovnc": "cmovae",
    "retn": "ret", "ret": "ret", "loopd": "loop", "enter": "enter",
    "movsd.rep": "rep movsd", "movsb.rep": "rep movsb", "movsw.rep": "rep movsw",
    "stosb.rep": "rep stosb", "stosd.rep": "rep stosd", "stosw.rep": "rep stosw",
    "scasb.repne": "repne scasb", "scasd.repne": "repne scasd",
    "cmpsb.repe": "repe cmpsb", "insd.rep": "rep insd", "outsd.rep": "rep outsd",
}

TAMANO = re.compile(
    r"\b(?:dword|word|byte|qword|tbyte|float|double|xmmword|ymmword|fpu|mmxword) ptr "
)
CADENA = re.compile(
    r"\b(?:rep|repne|repe)?\.?\s*\b(movs[bwd]|stos[bwd]|scas[bwd]|lods[bwd]"
    r"|cmps[bwd]|ins[bwd]|outs[bwd])\b"
)


def canonizar_cadena(texto: str) -> str | None:
    """`MOVSD.REP ES:EDI,ESI` y `rep movsd dword ptr …` -> `rep movsd`."""
    coincidencia = CADENA.search(texto)
    if not coincidencia:
        return None
    prefijo = "repne" if "repne" in texto else "rep"
    return f"{prefijo} {coincidencia.group(1)}"


def canonizar(texto: str) -> str:
    """Reduce una instrucción a una forma comparable entre Ghidra y Capstone."""
    t = texto.strip().lower()
    # Anotación de función contenedora, volcado de bytes en línea y comentario
    # de Ghidra (`PUSH 0x448250   ; "Terminal"`).
    t = t.split(" -> ", 1)[0].split(" | ", 1)[0].split(";", 1)[0]
    t = t.replace("\t", " ")
    t = re.sub(r"\s+", " ", t)
    t = t.replace(", ", ",").replace(" ,", ",").replace("[ ", "[").replace(" ]", "]")
    # Ghidra omite el prefijo de tamaño en algunos operandos de memoria.
    t = TAMANO.sub("", t)
    # Escalas: Ghidra escribe `*0x4` y explicita la escala 1 y el desplazamiento
    # nulo; Capstone escribe `*4` y los omite.
    t = re.sub(r"\*0x([0-9a-f]+)", r"*\1", t)
    t = re.sub(r"\*1(?=[\],\s]|$)", "", t)
    t = re.sub(r"\s*\+\s*0x0(?=[\],]|$)", "", t)
    t = re.sub(r"\[\s*0x0\s*\+\s*", "[", t)
    # Ghidra rellena los hexadecimales a 8 dígitos y los escribe en minúscula.
    t = re.sub(r"0x0*([0-9a-f]+)", r"0x\1", t)

    def negativo(coincidencia: re.Match) -> str:
        valor = int(coincidencia.group(2), 16)
        if valor >= 0x80000000:
            return f"{coincidencia.group(1)} - 0x{0x100000000 - valor:x}"
        return coincidencia.group(0)

    # Ghidra: `[ECX + 0xffffff00]`; Capstone: `[ecx - 0x100]`.
    t = re.sub(r"([a-z0-9_]+) \+ (0x[0-9a-f]+)", negativo, t)
    t = re.sub(r"\+ -", "- ", t)
    t = re.sub(r"\b(?:es|fs|gs|ds|ss):", "", t)
    # Registros de la FPU: `FSTP ST0`/`FXCH`/`FADDP` vs `fstp st(0)`/`fxch st(1)`.
    t = re.sub(r"\bst\((\d)\)", r"st\1", t)
    t = re.sub(r"^([a-z0-9.]+)\s+st0,", r"\1 ", t)
    if re.match(r"^fxch$", t):
        t = "fxch st1"
    if re.match(r"^f[a-z]+p$", t):
        t = f"{t} st1"
    t = t.strip()
    cabeza, _, resto = t.partition(" ")
    if cabeza in ALIAS:
        t = (ALIAS[cabeza] + " " + resto).strip()
    forma_cadena = canonizar_cadena(t)
    if forma_cadena is not None:
        return forma_cadena
    # Capstone imprime en decimal los inmediatos de un dígito; Ghidra, en hex.
    t = re.sub(r"(?<=[\[,+\-* ])0*(\d{1,3})(?=[\],]|\s|$)",
               lambda m: hex(int(m.group(1))), t)
    return t.strip()


def mnemonico(texto: str) -> str:
    """Cabecera canónica: dos tokens para las instrucciones de cadena."""
    partes = canonizar(texto).split(" ", 2)
    if partes and partes[0] in {"rep", "repne", "repe"} and len(partes) > 1:
        return f"{partes[0]} {partes[1]}"
    return partes[0].split(",")[0] if partes else ""


class Listado:
    """Contenido útil del volcado de Ghidrathon."""

    def __init__(self, ruta: str) -> None:
        self.instrucciones: dict[int, str] = {}
        self.conflictos: list[tuple[int, str, str]] = []
        self.volcados = 0
        self.etiquetas = 0
        self.rangos: list[tuple[int, int]] = []
        self.secciones: set[tuple[int, int]] = set()
        self.directivas: list[tuple[str, int, int | None]] = []
        self.contenedora: dict[int, int | None] = {}
        self.xr: dict[int, str] = {}
        self._leer(ruta)

    @staticmethod
    def _valor(anotacion: str) -> int | None:
        return int(anotacion[4:], 16) if anotacion.startswith("FUN_") else None

    def _leer(self, ruta: str) -> None:
        for cruda in open(ruta, encoding="utf-8", errors="replace"):
            linea = cruda.strip()
            if linea.startswith(">>>"):
                for tipo, inicio, fin in DIRECTIVA.findall(linea):
                    self.directivas.append(
                        (tipo, int(inicio, 16), int(fin, 16) if fin else None)
                    )
                continue
            rango = RANGO.search(linea)
            if rango:
                par = (int(rango.group(1), 16), int(rango.group(2), 16))
                self.rangos.append(par)
                self.secciones.add(par)
                continue

            direccion = None
            texto = ""
            anotacion: str | None = None
            instruccion = LINEA.match(linea)
            if instruccion:
                direccion = int(instruccion.group(1), 16)
                texto = instruccion.group(2)
                marcadas = ANOTACION.findall(linea)
                if marcadas:
                    # La anotación es la función contenedora de la PROPIA línea:
                    # `0x004099F5: CALL 0x0040af00 -> FUN_004099e0`.
                    anotacion = marcadas[-1]
            else:
                xr = XRLINEA.match(linea)
                variante = VARIANTE.match(linea)
                if xr:
                    direccion = int(xr.group(1), 16)
                    marcadas = ANOTACION.findall(linea)
                    anotacion = marcadas[-1] if marcadas else None
                    if anotacion is not None:
                        self.xr[direccion] = anotacion
                elif variante:
                    direccion = int(variante.group(1), 16)
                    anotacion = variante.group(2) or variante.group(3)
                    texto = variante.group(4).strip()
            if direccion is None:
                continue
            if anotacion is not None:
                self.contenedora.setdefault(direccion, self._valor(anotacion))
            if not texto:
                continue
            if VOLCADO.fullmatch(texto.split(" | ", 1)[0].split(";", 1)[0].strip()):
                self.volcados += 1
                continue
            if ETIQUETA_DATO.match(texto):
                self.etiquetas += 1
                continue
            anterior = self.instrucciones.get(direccion)
            if anterior is None:
                self.instrucciones[direccion] = texto
                continue
            if anterior != texto:
                self.conflictos.append((direccion, anterior, texto))
            # Ghidra imprime `None` donde no tiene instrucción; si otra parte del
            # volcado sí la lista, manda la instrucción real.
            if canonizar(anterior) == "none" and canonizar(texto) != "none":
                self.instrucciones[direccion] = texto

    @property
    def inicios_funcion(self) -> set[int]:
        return {v for v in self.contenedora.values() if v is not None}


def dueñas_por_flujo(analyzer: Analyzer) -> dict[int, set[int]]:
    """dirección -> funciones que la alcanzan por flujo."""
    duenas: dict[int, set[int]] = collections.defaultdict(set)
    for funcion in analyzer.functions.values():
        for instruccion in funcion.instructions:
            duenas[instruccion.address].add(funcion.address)
    return duenas


def dueñas_por_span(analyzer: Analyzer) -> dict[int, set[int]]:
    """dirección -> funciones cuyo span [inicio, fin] la contiene (criterio Ghidra)."""
    spans = sorted(
        (f.address, max(i.address for i in f.instructions))
        for f in analyzer.functions.values() if f.instructions
    )
    direcciones = sorted(analyzer.instructions)
    duenas: dict[int, set[int]] = collections.defaultdict(set)
    for inicio, fin in spans:
        for direccion in direcciones[bisect.bisect_left(direcciones, inicio):
                                    bisect.bisect_right(direcciones, fin)]:
            duenas[direccion].add(inicio)
    return duenas


def clase_de_diferencia(g: str, m: str, mg: str, mm: str) -> str:
    if mg != mm:
        return "mnemotécnico distinto"
    if re.sub(r"0x[0-9a-f]+", "0xN", g) == re.sub(r"0x[0-9a-f]+", "0xN", m):
        return "sólo difiere un inmediato/dirección"
    if g.replace(" ", "") == m.replace(" ", ""):
        return "sólo espaciado"
    if re.sub(r"-?0x[0-9a-f]+", "N", g) == re.sub(r"-?0x[0-9a-f]+", "N", m):
        return "forma del desplazamiento/inmediato"
    return "otra forma de operando"


def informe_listado(listado: Listado) -> None:
    unicos = len(listado.instrucciones)
    reales = [c for c in listado.conflictos
              if canonizar(c[1]) != canonizar(c[2])]
    print("== Listado de Ghidra")
    print(f"   secciones === ASM ===   : {len(listado.rangos)} "
          f"({len(listado.secciones)} rangos distintos, "
          f"{sum(b - a for a, b in sorted(listado.secciones))} bytes)")
    print(f"   direcciones con instrucción: {unicos}")
    print(f"   volcados de bytes       : {listado.volcados} líneas descartadas")
    print(f"   etiquetas de datos      : {listado.etiquetas} líneas (IAT/PTR_/DAT_)")
    print(f"   directivas D()/XR()     : {len(listado.directivas)} "
          f"({sum(1 for d in listado.directivas if d[0] == 'D')} D, "
          f"{sum(1 for d in listado.directivas if d[0] == 'X')} XR)")
    print(f"   direcciones listadas 2+ veces: {len(listado.conflictos)} "
          f"(sólo {len(reales)} difieren tras normalizar)")
    for direccion, antes, despues in reales[:4]:
        print(f"      {direccion:#010x}  {antes[:40]!r} != {despues[:40]!r}")
    print(f"   líneas XR()             : {len(listado.xr)}")
    print(f"   direcciones anotadas    : {len(listado.contenedora)} "
          f"({len(listado.inicios_funcion)} funciones citadas, "
          f"{sum(1 for v in listado.contenedora.values() if v is None)} sin función)")


def informe_cobertura(listado: Listado, mini: dict[int, object]) -> list[int]:
    ghidra = listado.instrucciones
    comunes = sorted(a for a in ghidra if a in mini)
    solas = sorted(a for a in ghidra if a not in mini)
    print("\n== Cobertura de direcciones")
    print(f"   en ambos                : {len(comunes)} "
          f"({100 * len(comunes) / len(ghidra):.2f}% del listado)")
    print(f"   sólo en Ghidra          : {len(solas)}")
    for direccion in solas[:8]:
        print(f"      {direccion:#010x}  {ghidra[direccion][:56]}")
    return comunes


def informe_instrucciones(listado: Listado, analyzer: Analyzer,
                          comunes: list[int], residual: str | None) -> None:
    mini = analyzer.instructions
    ghidra = listado.instrucciones
    crudo = normalizado = mismo_mnem = 0
    clases: collections.Counter = collections.Counter()
    ejemplos: dict[str, list[tuple[int, str, str]]] = collections.defaultdict(list)
    pares: collections.Counter = collections.Counter()
    diferencias: list[tuple[int, str, str]] = []
    for direccion in comunes:
        texto_ghidra = ghidra[direccion]
        texto_mini = f"{mini[direccion].mnemonic} {mini[direccion].op_str}".strip()
        if texto_ghidra.strip().lower() == texto_mini.lower():
            crudo += 1
        g, m = canonizar(texto_ghidra), canonizar(texto_mini)
        mg, mm = mnemonico(texto_ghidra), mnemonico(texto_mini)
        if mg == mm:
            mismo_mnem += 1
        else:
            pares[(mg, mm)] += 1
        if g == m:
            normalizado += 1
            continue
        diferencias.append((direccion, texto_ghidra, texto_mini))
        clase = clase_de_diferencia(g, m, mg, mm)
        clases[clase] += 1
        if len(ejemplos[clase]) < 4:
            ejemplos[clase].append((direccion, texto_ghidra, texto_mini))

    n = len(comunes)
    print(f"\n== Comparación instrucción a instrucción ({n} direcciones comunes)")
    print(f"   texto idéntico en crudo : {crudo} ({100 * crudo / n:.2f}%)")
    print(f"   idéntico tras normalizar: {normalizado} ({100 * normalizado / n:.2f}%)")
    print(f"   mismo mnemotécnico      : {mismo_mnem} ({100 * mismo_mnem / n:.2f}%)")
    if clases:
        print("\n   clases de diferencia residual:")
        for clase, cuenta in clases.most_common():
            print(f"     {clase:<38} {cuenta:6d}")
            for direccion, g, m in ejemplos[clase][:3]:
                print(f"        {direccion:#010x}  Ghidra: {g[:60]}")
                print(f"                    Mini  : {m[:60]}")
    if pares:
        print("\n   pares de mnemotécnicos distintos:")
        for (g, m), cuenta in pares.most_common(12):
            print(f"     Ghidra {g:<14} vs MiniGhidraPy {m:<14} {cuenta}")
    if residual:
        with open(residual, "w", encoding="utf-8") as salida:
            for direccion, g, m in diferencias:
                salida.write(f"{direccion:#010x}  Ghidra: {g.strip()}\n"
                             f"{'':<10}  Mini  : {m}\n")
        print(f"\n   {len(diferencias)} diferencias volcadas en {residual}")


def informe_funciones(listado: Listado, analyzer: Analyzer) -> None:
    flujo = dueñas_por_flujo(analyzer)
    span = dueñas_por_span(analyzer)
    anotadas = {d: f for d, f in listado.contenedora.items() if f is not None}
    print("\n== Función contenedora según las anotaciones del listado "
          f"({len(anotadas)} direcciones)")
    for criterio, duenas in (("span [inicio,fin] (Ghidra)", span),
                             ("pertenencia al flujo", flujo)):
        mismo = distinto = fuera = 0
        for direccion, inicio in anotadas.items():
            candidatas = duenas.get(direccion, set())
            if inicio in candidatas:
                mismo += 1
            elif candidatas:
                distinto += 1
            else:
                fuera += 1
        total = mismo + distinto + fuera
        print(f"   criterio {criterio:<28}: igual {mismo}/{total} "
              f"({100 * mismo / total:.1f}%) · distinta {distinto} · ninguna {fuera}")

    sin_funcion = [d for d, f in anotadas.items() if not span.get(d)]
    existe = sum(1 for d in sin_funcion if anotadas[d] in analyzer.functions)
    distinta = [d for d, f in anotadas.items()
                if span.get(d) and f not in span[d]]
    distinta_existe = sum(1 for d in distinta if anotadas[d] in analyzer.functions)
    print(f"   de las {len(sin_funcion)} sin función: en {existe} la función de "
          "Ghidra sí existe (cuerpo más corto)")
    print(f"   de las {len(distinta)} con función distinta: en {distinta_existe} "
          "la función de Ghidra sí existe (extensión/solape)")
    por_funcion = collections.Counter(anotadas[d] for d in sin_funcion)
    for funcion, cuenta in por_funcion.most_common(5):
        print(f"      {funcion:#x}: {cuenta} direcciones"
              + ("" if funcion in analyzer.functions else "  (función ausente)"))

    print(f"\n== Líneas XR() (dirección -> función contenedora): {len(listado.xr)}")
    for direccion, valor in sorted(listado.xr.items()):
        propias = sorted(span.get(direccion, set()))
        if valor.startswith("FUN_"):
            esperado = int(valor[4:], 16)
            estado = ("IGUAL" if esperado in propias else
                      ("sin función" if not propias else
                       "DISTINTA " + "/".join(f"{c:#x}" for c in propias)))
        else:
            estado = "IGUAL (sin función en ambos)" if not propias else (
                "DISTINTA " + "/".join(f"{c:#x}" for c in propias))
        print(f"   {direccion:#010x} -> {valor:<24} MiniGhidraPy: {estado}")

    citadas = listado.inicios_funcion
    propios = set(analyzer.functions)
    print(f"\n== Inicios de función citados por Ghidra: {len(citadas)}")
    print(f"   MiniGhidraPy también los tiene: {len(citadas & propios)} "
          f"({100 * len(citadas & propios) / len(citadas):.1f}%)")
    for inicio in sorted(citadas - propios):
        xrefs = analyzer.xrefs_to(inicio)
        tipos = collections.Counter(x.kind for x in xrefs)
        primera = analyzer.instructions.get(inicio)
        dentro = sorted(span.get(inicio, set()))
        print(f"   {inicio:#x}: xrefs={dict(tipos) or 'ninguno'} "
              f"MiniGhidraPy={'/'.join(f'{c:#x}' for c in dentro) or 'sin función'} "
              f"primera=`{primera.mnemonic} {primera.op_str[:26]}`"
              if primera else
              f"   {inicio:#x}: xrefs={dict(tipos) or 'ninguno'} sin instrucción")

    # ¿Las FUN_ citadas son hallazgos de Ghidra o arranques pedidos por el autor?
    inicios_rango = {a for a, _ in listado.secciones}
    print(f"\n   ¿oráculo independiente? {len(citadas & inicios_rango)}/{len(citadas)} "
          "inicios citados son arranque de un rango pedido; "
          f"{len(inicios_rango - citadas)} arranques de rango no se citan nunca")


def informe_codigo_huerfano(analyzer: Analyzer, listado: Listado) -> None:
    span = dueñas_por_span(analyzer)
    total = len(analyzer.instructions)
    fuera = sum(1 for a in analyzer.instructions if a not in span)
    print("\n== Código fuera de toda función")
    print(f"   instrucciones           : {total}")
    print(f"   fuera del span de toda función: {fuera} ({100 * fuera / total:.2f}%)")
    secciones = sorted(listado.secciones)
    dentro = sum(1 for a in analyzer.instructions
                 if a not in span and any(x <= a < y for x, y in secciones))
    print(f"      ...en zona que Ghidra sí listó: {dentro}")

    llamadas = 0
    explicadas: set[int] = set()
    orden = sorted(analyzer.instructions)
    posicion = {a: i for i, a in enumerate(orden)}
    huérfanas = {a for a in analyzer.instructions if a not in span}
    for direccion in orden:
        instruccion = analyzer.instructions[direccion]
        if instruccion.mnemonic != "call" or not instruccion.op_str.startswith("0x"):
            continue
        destino = int(instruccion.op_str.split()[0], 16)
        if destino not in analyzer.no_return_addresses:
            continue
        llamadas += 1
        siguiente = posicion[direccion] + 1
        while siguiente < len(orden) and orden[siguiente] in huérfanas:
            explicadas.add(orden[siguiente])
            siguiente += 1
    print(f"   llamadas a funciones no retornantes: {llamadas}")
    if llamadas:
        print(f"   huérfanas que siguen a una de esas llamadas: {len(explicadas)} "
              f"({100 * len(explicadas) / max(1, fuera):.1f}% de las que caen "
              "fuera de todo span)")

    dudosas = []
    for direccion in sorted(analyzer.no_return_addresses):
        funcion = analyzer.functions.get(direccion)
        if funcion and any(i.is_return for i in funcion.instructions):
            dudosas.append((direccion, len(funcion.instructions),
                            sum(1 for _ in analyzer.xrefs_to(direccion))))
    print(f"   marcas no-return cuyo cuerpo tiene `ret` (a auditar): {len(dudosas)} "
          f"de {len(analyzer.no_return_addresses)}")
    for direccion, cuerpo, xrefs in dudosas[:6]:
        print(f"      {direccion:#x}: {cuerpo} instrucciones, {xrefs} llamadas, "
              f"evidencia={analyzer.no_return_evidence.get(direccion, [])[:1]}")


def informe_rangos(listado: Listado, analyzer: Analyzer) -> None:
    unicos = sorted(set(listado.rangos))
    inicios = {a for a, _ in unicos}
    funcion = bloque = interior = sin_instruccion = 0
    for inicio in inicios:
        if inicio in analyzer.functions:
            funcion += 1
        elif inicio not in analyzer.instructions:
            sin_instruccion += 1
        else:
            try:
                propia = analyzer.function_at(inicio)
            except ValueError:
                interior += 1
                continue
            if any(b.start == inicio for b in getattr(propia, "blocks", ())):
                bloque += 1
            else:
                interior += 1
    print("\n== Rangos D(a,b) pedidos por el autor (no son fronteras de Ghidra)")
    print(f"   rangos únicos           : {len(unicos)} | inicios únicos: {len(inicios)}")
    print(f"   inicio de función       : {funcion} ({100 * funcion / len(inicios):.1f}%)")
    print(f"   inicio de bloque básico : {bloque} más")
    print(f"   dentro de otra función  : {interior}")
    print(f"   sin instrucción (datos) : {sin_instruccion}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("listado", help="volcado Ghidrathon de Ghidra (.txt)")
    parser.add_argument("binario", help="binario analizado por MiniGhidraPy")
    parser.add_argument("--residual", default=None,
                        help="escribe ahí todas las diferencias de texto")
    argumentos = parser.parse_args(argv)

    listado = Listado(argumentos.listado)
    informe_listado(listado)

    analyzer = Analyzer(BinaryImage.load(argumentos.binario))
    print("\n== MiniGhidraPy")
    print(f"   instrucciones           : {len(analyzer.instructions)}")
    print(f"   funciones               : {len(analyzer.functions)}")
    print(f"   thunks                  : {sum(1 for f in analyzer.functions.values() if f.is_thunk)}")

    comunes = informe_cobertura(listado, analyzer.instructions)
    informe_instrucciones(listado, analyzer, comunes, argumentos.residual)
    informe_funciones(listado, analyzer)
    informe_codigo_huerfano(analyzer, listado)
    informe_rangos(listado, analyzer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
