# MiniGhidraPy

Una herramienta pequeña, escrita en Python, para explorar estáticamente ejecutables **x86 de 32 bits y x86-64**. Está aislada del código Java original de Ghidra y sirve como prototipo/ayuda de aprendizaje; no es un reemplazo de Ghidra.

## Qué incluye

- Carga de ejecutables **ELF y PE** mediante LIEF y análisis de ficheros crudos con arquitectura/base indicadas. Mach-O x86 puede funcionar según la versión de LIEF, pero todavía no es una ruta de prueba garantizada.
- Detección automática de x86/x86-64 para ELF/PE; desensamblado Intel con Capstone.
- Resumen y secciones, símbolos, importaciones/exportaciones y cadenas ASCII/UTF-16LE.
- Descubrimiento heurístico de funciones (punto de entrada, símbolos, destinos de llamadas, firmas x86, inicios Mach-O `LC_FUNCTION_STARTS`, rangos `.pdata`/`UNWIND_INFO` de PE x64 y callbacks TLS de PE), thunks de salto directos/PLT, marcado de funciones conocidas `no-return` e inferencia conservadora por evidencias de llamadas/wrappers.
- Estimación de la purga de pila a partir de `ret`/`ret imm16` cuando los retornos recuperados coinciden; perfiles ABI y un motor de tipos/prototipos explícitos cubren convenciones x86/x86-64, clases de agregados, argumentos por referencia, retornos ocultos y promociones variádicas.
- Referencias cruzadas a llamadas/saltos, memoria y operandos escalares; detección conservadora de tablas de punteros a código y destinos CFG de saltos indirectos; traducción real SLEIGH/p-code por instrucción para x86/x86-64, además de pseudocódigo semántico experimental con perfiles ABI, propagación simbólica de registros/pila, expresiones aritméticas simplificadas, condiciones, uniones CFG mediante `phi(...)` y estructuración conservadora de `if`/`if-else` y `do/while` sencillos.
- Motor de impresión portado desde Ghidra (`Emit`, `EmitMarkup`, `EmitPrettyPrint`, `PrintLanguage` y `PrintC`): emisión directa de una función completa con resolución de símbolos/tipos, pretty printing de Oppen con ancho máximo configurable, recuperación de bucles `for` y markup `<clang_document>` con los elementos y colores originales.
- Capa de marshaling portada desde `marshal.hh`/`marshal.cc` (`XmlEncoder`, `XmlDecode`, `PackedEncoder`/`PackedEncode` y `PackedDecode`), con los registros de `ElementId`/`AttributeId`, los enumerados `OpCode` y `TypeMetatype`, y el modelo de documento *Clang* de la GUI de Ghidra (`ClangToken`, `ClangLine`, `ClangPrettyPrinter`, `ClangDocument`).
- Búsqueda de texto/bytes, etiquetas y comentarios guardados en un proyecto JSON, y parcheo seguro a un **archivo de salida** (el original no se modifica).
- Consola interactiva y una interfaz gráfica Tkinter opcional para recorrer funciones, cadenas, imports/exports, desensamblado, pseudocódigo, bloques e impresión `PrintC` (texto, markup, tokens coloreados y volcado `PackedEncode`).

## Instalación y uso

Requiere Python 3.10 o posterior. Desde esta carpeta:

```bash
python -m pip install -e .
minighidra ./programa
```

En Windows, si el ejecutable no está en `PATH`, también se puede ejecutar:

```powershell
python -m minighidra .\programa.exe
```

Ejemplos de comandos dentro de la consola:

```text
info
sections
functions
patterns
thunks
tables
unwind 0x140001000
noreturn
function main
strings password
xrefs 0x401000
disasm 0x401000 80
pseudo main
emit main
emit main --markup --width 100
emit main --packed --output main.packed
emit main --flat --labels join --hex
emit main --no-for-loops
graph main cfg.dot
rename 0x401000 mi_funcion
comment 0x401000 "validacion de entrada"
save proyecto.json
patch 0x401020 90 90 salida_patched.exe
help
```

También se puede abrir una interfaz gráfica sencilla basada en Tkinter o ejecutar una operación sin entrar en la consola:

```bash
minighidra ./programa --gui
minighidra ./programa --command "functions"
minighidra ./programa --command "disasm 0x401000 40"
```

Para un volcado de bytes sin cabeceras de ejecutable, indique arquitectura y dirección base:

```bash
minighidra firmware.bin --raw-arch x86_64 --base 0x400000
```

`patch DIRECCION HEX SALIDA` acepta bytes hexadecimales separados por espacios (por ejemplo, `patch 0x401020 90 90 copia.exe`). El rango debe pertenecer a bytes presentes en una sección y la salida no puede ser el archivo de entrada.

## Portado desde Ghidra

Esta iteración incorpora detección conservadora de prólogos x86/x86-64, inicios Mach-O `LC_FUNCTION_STARTS`, PE x64 `.pdata`/`UNWIND_INFO` y callbacks TLS, referencias escalares, tablas absolutas de punteros, resolución de saltos de tabla, catálogos `no-return`, purga de pila desde `ret`, thunks PLT, perfiles ABI, un registro explícito de tipos C, distribución tipada de prototipos, el prototipo de pseudocódigo semántico y el motor de impresión de Ghidra (`prettyprint.hh/.cc`, `printlanguage.hh/.cc`, `printc.hh/.cc`, `marshal.hh/.cc`, `metatype.hh`, `opcodes.hh`, `block.hh/.cc` y el modelo Java `ghidra.app.decompiler.Clang*`): la jerarquía `Emit` con markup XML/binario y pretty printing, la pila RPN de `PrintLanguage`, el back-end C `PrintC` con recuperación de bucles `for`, los codecs `XmlEncode`/`XmlDecode`/`PackedEncode`/`PackedDecode` y el documento *Clang* que consume la GUI, que emiten una función completa (cabecera, prototipo, declaraciones y cuerpo estructurado) sobre el AST del backend. Los modelos SysV AMD64 y Windows x64 contemplan agregados en registros/pila, sret y `varargs`; en x86 se cubren las reglas Cspec aplicables a los modelos cdecl, stdcall, fastcall, thiscall y regparm. Las fuentes revisadas, las reglas trasladadas y las diferencias con el motor original están documentadas en [`PORTING_NOTES.md`](PORTING_NOTES.md). El renderer textual de `SemanticDecompiler` emite las expresiones mediante una pila de tokens RPN propia (`PrintLanguageRpn`, en `minighidra/printlanguage_rpn.py`), con precedencias y paréntesis explícitos; `CEmitter(expression_mode="direct")` sigue disponible como alternativa para emitir desde el AST. Ambos caminos conviven: el AST C tipado alimenta el pseudocódigo del backend y el motor `Emit`/`PrintC` de Ghidra consume los artefactos estructurales de `decompile_artifacts()`.

## Tipos y prototipos x86/x86-64

El motor está en `minighidra/typesys.py` y `minighidra/prototype.py`. Registra tipos escalares C, typedefs, punteros, arrays, enums, vectores, structs y unions (incluidas etiquetas incompletas para referencias recursivas), además de tipos de función/punteros a función; calcula los offsets de campos y luego asigna cada parámetro/retorno a registros o pila según el perfil ABI. El resolvedor conserva las piezas de agregados, detecta `sret`, aplica promociones C para `...` y expone en la firma qué ubicación ocupa cada valor.

Se puede fijar un prototipo explícito antes de pedir pseudocódigo. Los tipos agregados se declaran en el registro asociado al analizador:

```python
from minighidra import Analyzer, FunctionParameter, FunctionSignature

analyzer = Analyzer(image)
record = analyzer.type_database.structure(
    "Record", [("tag", "uint32_t"), ("value", "double")]
)
function = analyzer.functions[0x401000]
signature = FunctionSignature(
    return_type="uint32_t",
    parameters=(FunctionParameter("item", record),),
    calling_convention="sysv_amd64",  # o __fastcall para Windows x64
)
resolved = analyzer.set_prototype(function, signature)
print(resolved.parameters[0].locations)
print(analyzer.pseudocode(function))
```

Para una función variádica, crea la firma con `variadic=True`. Si se conocen los tipos de los argumentos concretos de una llamada, `PrototypeResolver.resolve(..., variadic_types=(...))` devuelve sus ubicaciones después de las promociones C; en SysV también calcula el valor para `%al`. En x86, marca el parámetro oculto `this` con `implicit=True` cuando describas explícitamente un método. El resolvedor añade por separado el puntero oculto de retorno agregado cuando corresponde, sin incorporarlo a la lista de parámetros C visibles. Las ubicaciones y los tipos registrados se serializan en `Function.signature` y `export-json`.

Si la función no tiene una firma fijada, `candidate = analyzer.infer_prototype(function)` devuelve una propuesta **desbloqueada** (`source="inferred"`); si ya está fijada, la devuelve sin cambios. Sigue valores de entrada de registros y pila a través de la propagación simbólica; escrituras recientes a registros y slots salientes de pila en callsites pueden corroborar ubicaciones observadas o huecos posicionales anteriores. El análisis de pila es acotado y descarta recorridos al cruzar ramas o cambios de SP no modelados. `FunctionParameter.confidence` y `evidence`, junto con `FunctionSignature.inference_confidence` y `inference_evidence`, exponen por qué se propone cada parámetro. Son puntuaciones heurísticas de presencia/ubicación, no probabilidades calibradas; el tipo C puede seguir siendo un valor fallback. No se detectan formales nunca usados fuera de los huecos anteriores al último índice leído, y el orden entre bancos entero/SSE de SysV puede quedar ambiguo.

La traducción real de instrucciones a p-code SLEIGH está en `minighidra/sleigh.py`: `pypcode` ejecuta su backend nativo con archivos `.sla` compilados desde las definiciones locales de Ghidra 12.3 (`Ghidra/Processors/x86/data/languages/x86.slaspec` y `x86-64.slaspec`). `SleighTranslator(32|64)` devuelve por instrucción los opcodes ordenados y sus varnodes (espacio, offset, tamaño y representación), y expone en `.registers` una copia segura de alias/ubicaciones del lenguaje SLEIGH; las traducciones de la última decompilación están en `SemanticDecompiler.sleigh_instructions` y también se enlazan a `SemanticDecompiler.paramid_graph.instruction_flows` (`sleigh_ops`, `sleigh_error`). `sleigh_operation_count` cuenta estos ops nativos por separado.

Cuando no hay una ruta especializada para una instrucción, el renderer intenta consumir secuencias SLEIGH con operaciones enteras, booleanas y un subconjunto de `FLOAT_*` single/double (incluye `SQRT`, `LESS`, `ADD`, `MULT`, `TRUNC`, `INT2FLOAT` y `FLOAT2FLOAT`), registros/aliases de lanes XMM, temporales `unique` con subrangos solapados, flags, cargas/stores, varnodes RAM absolutos y ramificaciones internas acotadas (incluidos branches al fin de la instrucción). La bifurcación simbólica resuelve, entre otros casos, los bucles p-code de `BSF`/`BSR` x86/x86-64 y selecciones condicionales como `MINSS`; el pseudocódigo simbólico resume el conteo de bits con builtins. También se cubren `BSWAP`, `XCHG`, `ROL`, `SHLD`, `XADD` y operaciones bit a bit con memoria. Las rutas manuales de llamadas, retornos, x87 y otras instrucciones siguen teniendo prioridad; en secuencias ramificadas las cargas p-code se capturan en temporales y los stores se emiten bajo sus condiciones de camino; también se proyectan stores cuyo destino es un varnode RAM directo, lo que cubre `LOCK CMPXCHG qword ptr [rip+disp]` y permite reenviar/mezclar esa memoria en instrucciones posteriores. La pila abstracta conserva stores solapados por bytes en little-endian y recompone las lecturas posteriores, incluso al unir caminos del CFG. RAM reenvía valores solo para direcciones absolutas conocidas; un store por puntero no resuelto invalida RAM y la pila abstracta por posible aliasing. Las llamadas invalidan RAM y también la pila si reciben o conservan un puntero escapado a un local. Los userops `LOCK`/`UNLOCK` de las specs locales se conservan alrededor de operaciones atómicas lineales y ramificadas (`ADD`, `OR`, `CMPXCHG`, `XCHG`); también se modelan los userops locales `ROUNDPS`/`ROUNDSS`/`ROUNDPD`/`ROUNDSD` con modo inmediato constante 0–3; hay regresiones con fuentes RAM directas, varios lanes packed, signo de cero negativo y `ROUNDPS` en un CFG bifurcado. El modo de redondeo tomado de MXCSR (bit 2) y los demás `CALLOTHER` aún se rechazan. Las rutas que exceden los límites de caminos/pasos o usan opcodes/efectos no representables conservan el fallback `unsupported`. Esto integra una parte del p-code real en el pseudocódigo, pero no convierte la implementación en el pipeline completo de Ghidra.

```python
from minighidra import SemanticDecompiler
from minighidra.c_ast import CEmitter

decompiler = SemanticDecompiler(analyzer)
unit_ast = decompiler.decompile_ast(analyzer.functions[0x401000])
print(CEmitter(expression_mode="rpn").emit(unit_ast))
for instruction in decompiler.sleigh_instructions:
    print(hex(instruction.address), instruction.operation_names)

ssa = decompiler.ssa_function  # o analyzer.ssa(0x401000)
print(ssa.valid, ssa.complete, ssa.phi_count, ssa.memory_phi_count)
operation = next(op for op in ssa.operations if op.opcode == "INT_ADD")
print(decompiler.emit_pcode_operation(analyzer.functions[0x401000], operation.index))
print(decompiler.emit_pcode_operand(analyzer.functions[0x401000], operation.index, 0))
print(decompiler.emit_pcode_listing(analyzer.functions[0x401000]))
print(decompiler.emit_pcode_value_html(analyzer.functions[0x401000], operation.output_value))
```

El backend también puede recorrer las definiciones SSA directamente hasta los `SleighPcodeOp` y `SleighVarnode` originales; conserva coordenadas de p-code en los tokens, resuelve registros/parámetros y símbolos de direcciones disponibles, y ofrece markup HTML (`emit_html`). `emit_pcode_operation` admite dispatch por opcode y un fallback explícito `pcode_<opcode>(...)` para operaciones fuera del subconjunto. Ese fallback preserva operandos y origen, pero no inventa semántica C para un opcode desconocido.

`minighidra/ssa.py` implementa un núcleo SSA/heritage a partir de ese p-code real. Construye un CFG a nivel de microbloques que separa las ramas internas `BRANCH`/`CBRANCH`, calcula dominadores y fronteras, hace colocación de `phi` podada por vivacidad y renombra valores recorriendo el árbol de dominadores. Registra registros y temporales en celdas de byte para respetar solapamientos (`AL`/`AX`/`EAX`/`RAX`), una versión conservadora de memoria para `LOAD`/`STORE` y varnodes RAM directos, y efectos de llamada/retorno de registros según la convención ABI seleccionada. `SSAOperation.call_argument_values` conserva snapshots de los registros ABI de argumentos en cada callsite. `Analyzer.ssa(function_o_direccion)` ofrece el resultado directamente; `SemanticDecompiler.ssa_function` expone el mismo resultado de la última decompilación. `valid` informa de las invariantes/dominancia SSA y `complete` de errores de traducción o destinos de ramas p-code irresolubles; `return_sites` expone en cada `RETURN` los valores SSA observados en registros ABI candidatos a retorno. Nada de ello implica paridad con Ghidra.

El paso se contrastó con `heritage.cc/.hh` (`buildADT`, `calcMultiequals`, `renameRecurse`, `placeMultiequals`, `heritage`) y `block.cc` locales. Aún no porta los guards y pases iterativos completos de Ghidra: la memoria se modela con un token global (conservador, no por rangos/espacios), y faltan el refinamiento completo de rangos de varnode, guards de aliasing de LOAD/STORE y calls, joins de pila, propagación de tipos y otros pases de `HighFunction`.

`ParamIDAnalysis` mide entradas y retornos sobre el SSA p-code. El port reproduce los rangos de `ParamMeasure` por slot (branches/calls, `CALLOTHER`, `RETURN`, `INDIRECT` y lecturas ordinarias), recorre phis solo por aristas fuera de loop y limita el forward walk a `MAXDEPTH=10`. Para salidas implementa `walkbackward` y distingue `direct-write-without-read`, `direct-write-with-read`, `this-function-parameter`, `subfunction-return` e `indirect`; publica los resultados por registro en `output_measurements`. Los `LOAD`/`STORE` y los operandos de branch se puntúan como lecturas directas según `paramid.cc`, no como rank indirecto. `PcodeLikeGraph.from_ssa` conserva una operación por `SSAOperation`, phis, slices y sinks ABI. El grafo reducido complementa el naming de parámetros de pila y evidencia de argumentos/call-spec ausente del p-code SLEIGH sin prototipo. Como extensión consciente, `ROUND*` conserva el filtrado de lanes para no marcar bytes altos preservados como argumento escalar; las medidas de retorno se exponen por registro y combinan los return-sites alcanzables.

El modo `analyze(justproto=True, input_names=...)` limita medidas a los formales de una firma fijada, incluido el puntero sret oculto cuando existe; `SemanticDecompiler` lo usa para prototipos declarados y mide candidatos ABI/pila cuando infiere la firma. `ParamEntry` y `ParamActive` portan metadatos de recursos, clases, grupos, exclusiones, slots, rangos y estado de trials. `ParamListStandard` implementa `build_trial_map`, asignación, join/split, grupos de exclusión, holes de slots y cadenas inactivas; `ParamListRegister` hace el fill-in directo sin sintetizar holes. `PrimitiveExtractor` expande arrays/estructuras y refina las particiones primitivas comunes de unions; `join_dual_class` usa esa extracción para clasificar tiles, consumir registros/stack de forma transaccional y aplicar la justificación endian-aware de Ghidra. `JoinRecord` representa las piezas físicas y el tamaño lógico de direcciones unidas, incluido `extension="float"` (por ejemplo, un `float` lógico de 4 bytes respaldado por el registro `st0` de 10 bytes). Las listas de salida modelan join de retornos y sret. `ParamIDAnalysis.fillin_param_list` conecta los trials medidos de entrada y los registros de retorno observados con sus listas input/output; `PrototypeModel`/`ProtoModelMerged` leen las reglas Cspec locales x86/x86-64, asignan firmas fijadas y puntúan modelos candidatos. El alcance exacto y las diferencias restantes se detallan en [`PORTING_NOTES.md`](PORTING_NOTES.md): no se declara paridad total; quedan reglas de modelo, formatos de tipos y pases de decompilación sin portar.

Con una firma fijada, el pseudocódigo usa los tipos/nombres y ubicaciones ABI para sus parámetros, modela argumentos agregados directos cuando están en registros y no imprime el puntero ABI oculto de un `sret` como argumento de fuente. La recuperación de los tipos efectivos de argumentos `...` desde binario y la materialización de todos los retornos agregados en el cuerpo siguen siendo aproximaciones.

## Motor de impresión (Emit/PrintC)

`minighidra/emit.py`, `minighidra/printlanguage.py`, `minighidra/printc.py` y `minighidra/printdoc.py` portan el motor de impresión del decompiler de Ghidra. La emisión es directa y cubre la función completa: comentarios de cabecera, prototipo (tipo de retorno, convención, nombre y parámetros), declaraciones de locales y cuerpo estructurado (`if`/`else`, `while`, `do/while`, `switch`, `goto`/etiquetas), todo a través de la pila RPN de `PrintLanguage` y del algoritmo de pretty printing de Oppen.

El backend normal del decompiler **sigue usando su AST**: `SemanticDecompiler` continúa propagando valores y construyendo expresiones `_Expr`, y `pseudo` no cambia. `decompile_artifacts()` expone de forma aditiva la firma resuelta, las sentencias por bloque con su dirección de instrucción, los terminadores estructurados (condición, destinos, casos de switch, tail calls), las llamadas con sus argumentos y el grafo colapsado; `printdoc.py` resuelve con eso cada símbolo (parámetro, local, global) y cada tipo contra el registro C del analizador.

```python
from minighidra import Analyzer, BinaryImage

analyzer = Analyzer(BinaryImage.load("./programa"))
function = analyzer.functions[0x401000]

print(analyzer.print_function(function))                      # texto con pretty printing
print(analyzer.print_function(function, markup=True))         # XML <clang_document>
print(analyzer.print_function(function, markup=True, packed=True))   # bytes PackedEncode
print(analyzer.print_function(function, max_line_size=72))    # ajusta a 72 columnas
print(analyzer.print_function(function, flat=True))           # etiquetas + goto, sin estructurar
print(analyzer.print_function(function, for_loops=False))     # while/do-while, sin `for`
print(analyzer.print_function(function, integer_format="hex", comment_style="cplusplus"))

documento = analyzer.clang_document(function)                 # modelo Clang de la GUI
print(documento.text())                                       # idéntico a print_function()
print(documento.signature())                                  # prototipo desde <funcproto>
for texto, color in documento.spans():                        # tokens + color de resaltado
    print(repr(texto), color)
```

En consola:

```text
emit main
emit main --markup --output main.xml
emit main --packed --output main.packed
emit main --width 72 --indent 4
emit main --flat --labels join
emit main --no-casts --hex --comments cplusplus --brace same_line
emit main --no-for-loops
```

`emit --packed` implica `--markup`: con `--output` escribe los bytes del `PackedEncode` y sin él imprime el tamaño y un volcado hexadecimal.

El markup reproduce el vocabulario original de `EmitMarkup`: `<clang_document>`, `<function>`, `<funcproto>`, `<return_type>`, `<vardecl>`, `<block blockref>`, `<statement opref>`, `<variable varref opref color>`, `<op>`, `<funcname>`, `<type id>`, `<field name off>`, `<comment space off>`, `<label>`, `<value>`, `<syntax open|close>` y `<break indent>`, con los colores de `syntax_highlight` (parámetros `0x6`, variables `0x4`, globales `0x7`, tipos `0x2`, funciones `0x3`, constantes `0x5`, comentarios `0x1`). Las opciones del emisor están expuestas en `PrintC` (`set_no_cast_printing`, `set_integer_format`, `set_comment_style`, `set_brace_format_*`, `set_flat`, `set_markup`, `set_max_line_size`, `set_indent_increment`, `set_label_mode`, `set_analyze_for_loops`) y coinciden con las de Ghidra.

### Marshaling: XML y `PackedEncode`

`minighidra/marshal.py` porta `marshal.hh`/`marshal.cc`. `XmlEncoder` (alias `XmlEncode`) escribe el markup XML y `XmlDecode` lo vuelve a leer; `PackedEncoder` (alias `PackedEncode`) y `PackedDecode` implementan el formato binario compacto de Ghidra byte a byte: cabeceras de elemento/atributo de uno o dos bytes con byte de extensión, enteros con código de longitud y 7 bits de datos por byte, cadenas UTF-8 con prefijo de longitud y ningún byte nulo. Los registros de ids (`ElementId`/`AttributeId`, incluidos `ELEM_UNKNOWN`=292 y `ATTRIB_UNKNOWN`=161), el enumerado `OpCode` de `opcodes.hh` y `TypeMetatype` de `metatype.hh` viven en el mismo módulo. `set_packed_output(True)` activa el codec binario y `Emit.get_packed_bytes()` devuelve los bytes; `PackedDecode`/`XmlDecode` reconstruyen el mismo árbol, así que el markup puede verificarse por ida y vuelta. En las muestras de prueba el formato binario ocupa entre el 25 % y el 30 % de los bytes del XML (25 funciones de dos binarios ELF x86-64, árbol decodificado idéntico al del XML en todos los casos).

### Modelo de documento *Clang*

`minighidra/clangdoc.py` porta el consumidor de la GUI de Ghidra (`ghidra.app.decompiler.Clang*`, `ClangMarkup.buildClangTree`, `DecompilerUtils.toLines` y `PrettyPrinter`): `build_clang_tree()` decodea XML **o** bytes `PackedEncode` en un árbol de `ClangNode`/`ClangToken`, `to_lines()` lo parte en `ClangLine` por cada `ClangBreak` y `ClangPrettyPrinter`/`ClangDocument` devuelven el texto, las líneas, los pares `(texto, color)` y la firma. `Analyzer.clang_document()` encadena emisión y decodificación, y la pestaña **Impresión (PrintC)** de `gui.py` muestra el texto, el markup, los tokens coloreados y el volcado binario con las opciones de ancho, indentación, casts, formato de enteros, comentarios, llaves, etiquetas y bucles `for`.

### Bucles `for` y tokens opacos

`PrintC.emit_for_loop` reproduce `BlockWhileDo::finalTransform`: cuando un `while` tiene una cabecera cuya única variable `phi` alimenta la condición y el bloque final sólo vuelve a la cabecera, el bucle se imprime como `for (inicialización; condición; incremento)`. El resto de `phi` de la cabecera no se mezcla en la cabecera del bucle: se aplazan y se imprimen como primeras sentencias del cuerpo. `set_analyze_for_loops(False)` (o `emit --no-for-loops`) desactiva la transformación y vuelve al `while`/`do-while` estructural.

Las expresiones que el backend aún no modela (expansiones SIMD/MMX por lanes, bit-punning con uniones, paridad, agregados) se emitían como un único blob `<syntax>` sin colores. Ahora el texto opaco se segmenta en átomos por sus operadores y puntuaciones y se emite como varios `<syntax>` unidos por un `OpToken` asociativo: el texto resultante es idéntico byte a byte, pero cada fragmento recibe su color, puede partirse entre líneas y el conteo de nodos opacos de las muestras de punto flotante baja de 22 a 14 y de 8 kinds a 2 (`simd_float` y `simd_convert_fp`). `printdoc.py` modela además las cargas/ops/comparaciones/casts/condiciones x87 y `concat` como nodos reales (`PrintExpr`), con nombres libm (`sqrtl`, `nearbyintl`, `atan2l`, `fmodl`) y `isnan`/`isunordered` para las condiciones.

## Límites conocidos

Esto es una **primera versión funcional**, no tiene paridad con Ghidra. El pseudocódigo aplica un subconjunto de decompilación semántica (propaga registros y slots de pila, simplifica operaciones enteras, expresa condiciones de flags y resuelve algunos `phi(...)` de unión como asignaciones dentro de `if`/`if-else`). El estructurador acepta diamantes forward sencillos con brazos privados de un bloque y join común, ramas privadas que terminan en retorno/trap, y el patrón `do/while` de un bloque con una arista a sí mismo y salida forward. En los bucles conserva los `phi` internos, sin normalizar variables de iteración; los `while` con cabecera/cuerpo separados, loops de varios bloques, `break`/`continue`, backedges complejos, destinos compartidos, saltos irregulares o merges no demostrables conservan etiquetas y `goto` como fallback. Deja comentarios para instrucciones no soportadas. El fallback SLEIGH consume un subconjunto de ops enteros, booleanos y float single/double, registros/flags/memoria y algunas ramificaciones internas acotadas (incluidos los bucles p-code de `BSF`/`BSR` y selecciones como `MINSS`). El núcleo SSA/heritage descrito arriba procesa operaciones p-code y sus ramas, pero no implementa todos los opcodes ni los guards, joins de pila, refinamientos de rangos, análisis de aliasing/calls ni pases iterativos del motor C++. El registro C y el resolvedor de prototipos cubren layouts ABI concretos, pero no la recuperación de tipos desde DWARF/PDB, aliasing/bitfields completos ni la inferencia completa de parámetros; la inferencia automática sigue siendo heurística. La precedencia/emisión de expresiones del renderer del backend usa un port parcial de la pila RPN de `PrintLanguage`/`PrintC` (`printlanguage_rpn.py`), y hay además una ruta directa de inspección/emisión por operación y valor que recorre el SSA enlazado a `PcodeOp`/`Varnode`; no sustituye aún el renderer de la función completa. El motor de impresión de Ghidra (`Emit`/`PrintLanguage`/`PrintC`), la capa de marshaling (`XmlEncode`/`XmlDecode`/`PackedEncode`/`PackedDecode`) y el modelo de documento *Clang* sí están portados y emiten una función completa con markup, pretty printing y bucles `for`, pero trabajan sobre el AST del backend: las expansiones SIMD/MMX por lanes, el bit-punning con uniones, la paridad y los agregados siguen siendo texto opaco (ahora segmentado en átomos coloreados y ajustables, pero sin nodos reales), los casts provienen de los anchos/signos del AST y no de la propagación de tipos de Ghidra, `MULTIEQUAL` continúa imprimiéndose como `phi(...)` porque no hay pase de heritage en el motor de impresión (sólo el `phi` que hace de variable de bucle se reparte entre inicialización e incremento), no se resuelven tipos en los tokens `<type>`/`<field>` del documento *Clang*, y no hay `EquateSymbol`, bitfields, árbol de namespaces ni `CommentSorter` completo. La resolución de símbolos usa la información disponible en MiniGhidraPy, el markup HTML aporta clases semánticas propias y los opcodes no cubiertos conservan un fallback genérico, no la semántica específica de todos los fallbacks de Ghidra. Por tanto no se promete pseudocódigo idéntico al de Ghidra ni semántica completa para cualquier instrucción. Tampoco hay depurador, emulación, análisis dinámico, scripting/plugin API, colaboración ni soporte de otras arquitecturas. El desensamblador está limitado a x86 de 32 bits y x86-64. La importación de formatos depende de las capacidades de LIEF instalado.

## Pruebas

```bash
python -m unittest discover -s tests -v
```
