# Notas de portado desde el source de Ghidra

## Port de este incremento: identificación de inicios de función x86

Se revisaron estas implementaciones y datos del repositorio:

- `Ghidra/Features/BytePatterns/src/main/java/ghidra/app/analyzers/FunctionStartAnalyzer.java`
- `Ghidra/Processors/x86/data/patterns/x86gcc_patterns.xml`
- `Ghidra/Processors/x86/data/patterns/x86-64gcc_patterns.xml`
- `Ghidra/Processors/x86/data/patterns/x86win_patterns.xml`

`FunctionStartAnalyzer` usa patrones de bytes, restricciones de código precedente/posterior y, por defecto, limita la búsqueda a bloques ejecutables. `function_starts.py` traslada esa idea a una tabla reducida de prólogos GCC/Windows comunes: el prólogo debe coincidir en una dirección de instrucción decodificada, estar en una sección ejecutable y cumplir un límite de flujo previo (inicio de sección, `ret`, salto incondicional, relleno o trap). También se trasladó el caso Windows `sub esp, imm8` después de un límite, exigiendo al menos dos instrucciones válidas para reducir falsos positivos. Las firmas genéricas de `sub rsp, ...` mantienen un requisito mayor de código alcanzable. Los candidatos aceptados aparecen en `patterns` y como origen de función en `functions`.

No se copió el motor de patrones de Ghidra: **no** se implementan `PatternDecisionTree`, el lenguaje completo de máscaras XML, los contextos de registros, las restricciones `after=...`, la gestión de bookmarks ni el `PseudoDisassembler` de Ghidra. La tabla Python es una selección explícita, no una traducción exhaustiva de los XML.

## Thunks y resolución de PLT

Este incremento también revisa `Ghidra/Features/Base/src/main/java/ghidra/app/cmd/function/CreateThunkFunctionCmd.java` (`getThunkedAddr`, `getSimpleFlow`) y `Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/function/FunctionAnalyzer.java`. Se reconocen saltos incondicionales directos con prólogo NOP/ENDBR opcional, y stubs ELF PLT que saltan por una ranura GOT. Para ELF se usa el orden de `pltgot_relocations` de LIEF y se valida el tamaño/stride de `.plt` o `.plt.sec` antes de crear nombres de stub; los layouts no habituales se ignoran.

El detector no emula el chequeo p-code de hasta ocho instrucciones ni el análisis de registros/efectos laterales de Ghidra. Tampoco resuelve thunks calculados por registros ni toda la variedad de referencias indirectas. La consola los muestra con `thunks`; el grafo del caller y las referencias pueden resolver los nombres PLT y las llamadas importadas no-return.

## Inicios de función Mach-O

Este subconjunto se basa en `Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/analysis/MachoFunctionStartsAnalyzer.java` y `Ghidra/Features/Base/src/main/java/ghidra/app/util/bin/format/macho/commands/FunctionStartsCommand.java`. Se leen los offsets ULEB128 de `LC_FUNCTION_STARTS` mediante LIEF, se suman a la dirección virtual del segmento `__TEXT` y se consideran después de recuperar las funciones evidentes, siguiendo la prioridad tardía de Ghidra. Solo se añaden como semillas si apuntan a una instrucción decodificada en bytes ejecutables, no son traps y todavía no pertenecen a un cuerpo recuperado. `info`, `functions` y `export-json` permiten ver los resultados.

Ghidra advierte que la lista puede contener falsos inicios; el port no replica `PseudoDisassembler.isValidSubroutine`, bookmarks ni el análisis de dyld caches divididas. Se limita a los comandos que LIEF exponga para la slice Mach-O elegida por el cargador (en binarios universales se prefiere x86-64); un punto que decodifica como instrucción aún puede ser un falso positivo.

## Inicios x64 de PE desde `.pdata`

Este subconjunto se basa en `Ghidra/Features/Base/src/main/java/ghidra/app/util/bin/format/pe/ExceptionDataDirectory.java` y `ImageRuntimeFunctionEntries_X86.java` (la tabla de 12 bytes `RUNTIME_FUNCTION` usada por PE x64). Se usan las entradas `exception_functions` de LIEF cuando están disponibles; si no, se lee el directorio EXCEPTION_TABLE desde las secciones PE y se decodifican las triples little-endian de `.pdata`. Se suman las RVA a la image base y se crean semillas en `BeginAddress`, salvo entradas encadenadas (`UNW_FLAG_CHAININFO`), rangos malformados o RVAs nulos. Se preservan los rangos en JSON y el origen se ve en `functions`.

El port solo aplica a PE x86-64 y no implementa las tablas de excepción de ARM/CHPE. Las funciones solo se crean si el inicio cae en código ejecutable decodificado.

## Metadatos x64 `UNWIND_INFO`

Siguiendo `Ghidra/Features/Base/src/main/java/ghidra/app/util/bin/format/pe/PEx64UnwindInfo.java`, se decodifican versión/flags, tamaño del prólogo, registro de marco, slots de `UNWIND_CODE` y RVA del handler cuando existe. Se reconocen los códigos habituales de push, asignación de pila, establecimiento de frame y guardado de registros; de ellos se estima `stack_allocation_bytes` y se exponen los registros guardados junto con las operaciones. El resultado se adjunta a cada función PE x64 y se exporta en JSON; `functions` muestra la asignación estimada cuando se puede calcular.

No se ejecuta el algoritmo de unwinding ni se interpreta el lenguaje de los exception handlers o los payloads de chained unwind. Los códigos desconocidos/insuficientes dejan la estimación de pila como desconocida; la cifra no reemplaza el análisis formal del stack frame de Ghidra.

## Callbacks TLS de PE

El port se basa en `Ghidra/Features/Base/src/main/java/ghidra/app/util/bin/format/pe/TLSDataDirectory.java` y `TLSDirectory.java`. Se toman de LIEF las direcciones de `TLS.callbacks` (con normalización VA/RVA), se limita la lectura a 20 como en Ghidra y se añaden como semillas con nombres `tls_callback_N`; solo se recuperan si caen en instrucciones de secciones ejecutables. `info` y `export-json` exponen la lista.

No se marcan el array de punteros ni `_tls_index`, ni se recrean bookmarks/referencias de datos. La validez final de cada callback depende del filtro de código ejecutable del analizador.

## Llamadas `call` al fall-through

`Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/function/FunctionAnalyzer.java` omite como inicio de función un destino de llamada que coincide con la instrucción fall-through. El port hace lo mismo para patrones x86 `call $+5` usados para obtener la dirección de retorno en thunks PC-relativos; mantiene el xref de llamada, pero no crea una función ficticia en el byte siguiente.

## Funciones conocidas que no retornan

El siguiente port está basado en `Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/analysis/NoReturnFunctionAnalyzer.java`, `NonReturningFunctionNames.java` y `Ghidra/Features/Base/data/noReturnFunctionConstraints.xml`. Se incluyen las listas de nombres ELF, PE, Mach-O, Go y Rust de `Ghidra/Features/Base/data/*FunctionsThatDoNotReturn` en `minighidra/data/noreturn/`.

Se reproduce el filtrado por nombre exacto/prefijo con `*`, el descarte de `_` iniciales y la selección de listas por formato. Como MiniGhidraPy no detecta de forma fiable el compilador, añade los nombres Go/Rust a las listas del formato correspondiente; esto puede marcar una coincidencia que Ghidra solo consideraría para ese compilador. Los nombres de símbolos con versión ELF y sufijo `@plt` se normalizan antes de comparar.

Las funciones reconocidas quedan marcadas `no_return`; si una llamada directa o indirecta apunta a una de ellas, el recuperador corta el flujo del llamador en esa instrucción, evitando incorporar el fall-through inalcanzable. La consola lo muestra con `noreturn`. No se portan bookmarks, namespaces, reparación de flujos ni importación de funciones externas.

## Inferencia heurística de no-return

Este incremento traslada un subconjunto de `Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/analysis/FindNoReturnFunctionsAnalyzer.java`. Se agrupan llamadas por destino y se consideran indicios un inicio de función, una instrucción no decodificada o un trap en la cadena de fall-through. Ghidra comprueba explícitamente `INT3` para x86; este port también trata `ud2` y `hlt` como traps. El recorrido Python está limitado a 256 instrucciones para evitar escaneos descontrolados. De forma predeterminada hacen falta tres sitios de llamada con indicios para marcar el destino; el umbral se puede ajustar con `Analyzer(image, noreturn_evidence_threshold=N)`.

Después, la propiedad se propaga conservadoramente a wrappers cuyos caminos terminales recuperados llaman o saltan únicamente a funciones no-return. La propagación iterativa resuelve cadenas de wrappers y, al finalizar, se reconstruyen los cuerpos una vez para cortar el flujo de sus callers.

No se reproduce el modelo de referencias/listing de Ghidra: la asociación de destinos depende de las referencias que Capstone/LIEF permitan resolver y de los inicios de función ya recuperados. Quedan fuera `callfixup`, referencias inconsistentes a datos/código, bookmarks, creación/reparación de instrucciones y cuerpos exactos del CFG de Ghidra. La prueba de wrappers usa el CFG heurístico de MiniGhidraPy; no debe tomarse como garantía formal de que todos los caminos de ejecución terminan.

## Referencias de operandos escalares

Este port parcial se basa en `Ghidra/Features/Base/src/main/java/ghidra/app/plugin/core/analysis/ScalarOperandAnalyzer.java`. Para formatos no ELF con una base distinta de cero, se consideran referencias escalares los inmediatos no-control-flow que apuntan a una sección mapeada, ignorando valores pequeños y constantes centinela. Se agregan como xrefs `scalar`; también aparecen en el desensamblado y en `export-json`. No se crean funciones a partir de esos datos.

A diferencia de Ghidra, no se consultan relocalizaciones para validar escalares, no se resuelven espacios de direcciones alternativos ni se analizan tablas de salto/subrutinas. Para limitar falsos positivos, los destinos ejecutables solo se conservan si ya son un inicio de función o tienen símbolo; los destinos de secciones no ejecutables se conservan directamente.

## Tablas de punteros y saltos indirectos

Este subconjunto se basa en `OperandReferenceAnalyzer.getAddressTable` y `AddressTable.getEntry`. Si una instrucción referencia una dirección de una sección no ejecutable, se comprueba si allí empieza una secuencia de al menos tres punteros absolutos consecutivos (tamaño 4 u 8 bytes según x86/x86-64) que resuelven a instrucciones decodificadas de secciones ejecutables. Se registran los destinos con xrefs `table`; `tables` y `export-json` muestran las tablas. También se conserva la base de una dirección indexada sin registro base, como `jmp [eax*4 + tabla]`. Si un `jmp` indirecto referencia exactamente una tabla reconocida, sus destinos se enlazan como sucesores del CFG, se marcan con xrefs `switch` y se recuperan esos bloques.

Es una aproximación: solo se examinan direcciones ya referenciadas, se limita cada secuencia a 512 entradas y no se interpretan relocalizaciones, tablas de saltos relativas, entradas con ajustes ni información de tipo. No se enlazan tablas a llamadas indirectas y las referencias a la tabla no crean funciones automáticamente.

## Purga de pila en retornos x86

A partir de `Ghidra/Features/Base/src/main/java/ghidra/app/cmd/function/FunctionPurgeAnalysisCmd.java`, se infiere `stack_purge_size` desde las instrucciones de retorno alcanzables: `ret` equivale a cero y `ret imm16` usa el inmediato. Solo se publica un valor si todos los retornos decodificados coinciden; la consola y `export-json` lo exponen. El valor conserva el indicio directo de limpieza por el callee; por sí solo no determina la convención, aunque el resolvedor ABI posterior lo combina con decoración y evidencia de los callsites.

No se implementa el análisis de referencias a la pila, la creación de stack frame, el modelo de parámetros ni la selección de convenciones near/far de Ghidra. Un valor cero tampoco distingue por sí solo entre cdecl y otras convenciones, y los retornos no recuperados o contradictorios quedan sin valor.

## Recuperación del cuerpo de función

Como referencia de flujo se consultaron `Ghidra/Features/Base/src/main/java/ghidra/app/cmd/function/CreateFunctionCmd.java` y su uso de `FollowFlow`. El análisis Python sigue el flujo condicional/directo interno y el fall-through, se detiene en retornos/saltos terminales y no sigue destinos de llamadas. Evita incluir inicios ya conocidos para reducir solapamientos.

Esto reproduce la idea general de `getFunctionBody`, pero no el modelo de memoria/listing, las referencias indirectas, la resolución de thunks, la reconciliación de cuerpos solapados ni la búsqueda hacia atrás de `findFunctionEntry`.

## Referencias de código

Las firmas viven en `minighidra/function_starts.py`; el análisis no-return en `minighidra/noreturn.py` y `analysis.py`; las tablas/CFG de saltos en `analysis.py`; la resolución de thunks PLT en `image.py` y `analysis.py`; las listas de nombres están en `minighidra/data/noreturn/`. La CLI ofrece `patterns`, `thunks`, `tables` y `noreturn`; `export-json` incluye indicios `no_return_evidence`, tablas y destinos computados.

## Convenciones de llamada y firmas x86/x86-64

El subconjunto ABI está en `minighidra/abi.py` y se conecta al pseudocódigo en `minighidra/decompiler.py`. Se contrastó con los Cspec locales `Ghidra/Processors/x86/data/languages/x86win.cspec`, `x86gcc.cspec`, `x86-64-win.cspec` y `x86-64-gcc.cspec`:

- PE x86 usa por defecto `__stdcall`; ELF/raw x86, `__cdecl`. También se modelan `__fastcall` (ECX, EDX y luego pila), `__thiscall` de Windows (ECX como `this` oculto en la expresión de llamada, pero conservado como primer parámetro formal), y `__regparm1/2/3` de GCC (EAX, EDX, ECX y luego pila). Se reconoce la decoración MSVC habitual y se infieren casos sin decoración con lecturas de registros de entrada, escrituras recientes en los callsites y limpieza observada de pila (`ret imm16` frente a `add/lea esp` o `pop`).
- PE x64 usa el perfil `__fastcall` posicional de Windows: RCX/RDX/R8/R9, alternados por posición con XMM0–XMM3 para argumentos flotantes, shadow space de 32 bytes y argumentos apilados desde el offset 40 relativo al RSP de entrada. Los métodos MSVC decorados pueden usar el mismo perfil con `this` oculto en RCX.
- El Cspec x86-64 GCC de Ghidra llama `__stdcall` a su modelo SysV AMD64: XMM0–XMM7 para valores flotantes, RDI/RSI/RDX/RCX/R8/R9 para enteros, después pila; retornos enteros en RAX/RDX y flotantes en XMM0/XMM1. El perfil Python se etiqueta `sysv_amd64` para distinguirlo del ABI PE.
- Los perfiles describen registros de argumento, pila, retorno, caller-/callee-saved y limpieza. Las llamadas directas consumen el perfil del destino; los argumentos `this` ocultos no se imprimen como argumento explícito. El decompilador conserva las ubicaciones de retorno EAX/EDX:EAX, RAX/RDX, ST0 y XMM0 en las rutas implementadas.

Tras decompilar una función, `Function.calling_convention` y `Function.signature` exponen el modelo inferido; `FunctionSignature` incluye tipo de retorno y parámetros con los tipos heurísticos recuperados del cuerpo. `export-json` serializa esos campos cuando ya se calculó la firma. Los tipos no observables se dejan como enteros del ancho de arquitectura. La eliminación de un retorno escalar para declarar `void` y la inferencia de resultados anchos/vectoriales dependen de evidencia de los callers; con xrefs irresolubles o flujo ambiguo no se descarta el valor observado.

No se portan aún la resolución completa `resolveprototype`, el cálculo de clases/join para estructuras y argumentos agregados, los retornos ocultos por puntero, `varargs`, `syscall`, `processEntry` ni todas las variantes GCC/MSVC. Tampoco se recuperan metadatos de tipos o clases C++; el orden de parámetros mixtos SysV y las firmas de funciones sin xrefs pueden quedar incompletos. Por tanto, son modelos de ABI y firmas parciales, no una reproducción íntegra del motor de prototipos de Ghidra.

## Primer prototipo de pseudocódigo semántico

El prototipo vive en `minighidra/decompiler.py` y se conecta desde `Analyzer.pseudocode`. Se revisaron en este checkout:

- `Ghidra/Features/Decompiler/src/main/java/ghidra/app/decompiler/DecompInterface.java`: el decompilador Java envía la solicitud a un proceso persistente, comprueba el resultado y recibe el árbol/markup y el C. MiniGhidraPy aún no invoca ese proceso: mantiene el requisito de una implementación Python.
- `Ghidra/Features/Decompiler/src/decompile/cpp/coreaction.cc`: `ActionDatabase::universalAction` muestra las fases de heritage, limpieza de código muerto, recuperación de tipos, recuperación de bloques y merges de varnodes.
- `Ghidra/Features/Decompiler/src/decompile/cpp/heritage.cc` y `op.hh`: referencia del modelo de variables/p-code SSA y de los merges de valores en uniones del CFG.
- `Ghidra/Features/Decompiler/src/decompile/cpp/blockaction.cc` / `blockaction.hh`: `ActionBlockStructure` y clases de bloques estructurados (`BlockIf`, `BlockWhileDo`, etc.); el primer prototipo conserva `goto` y no porta este estructurador.
- `Ghidra/Features/Decompiler/src/decompile/cpp/opcodes.hh` y `printc.cc`: referencia para operadores enteros y su impresión como expresiones C.

La implementación Python vuelve a decodificar cada instrucción con Capstone en detalle y traduce un subconjunto explícito a expresiones simbólicas. Propaga registros de propósito general, MMX de 64 bits, XMM de 128 bits, pila y una pila x87 abstracta por el CFG; crea `phi(...)` en uniones y pliega constantes. Modela, entre otras, operaciones aritméticas/lógicas, flags y condiciones comunes, `ADC`/`SBB`, `MUL`/`IMUL`/`DIV`/`IDIV` implícitos, transferencias x87 de comparación a `SAHF`, movimientos x87; un subconjunto MMX; y movimientos SSE, operaciones empaquetadas de enteros (`PUNPCKLDQ`, `PSHUFD`, `PCMPEQD`, `PMINSD`/`PMAXSD`, shifts y lógica) y conversiones/operaciones escalares SSE de punto flotante. También modela `MOVS`/`STOS`/`LODS`/`CMPS`/`SCAS` con prefijos `REP*`, el contador y los punteros, y el sentido conocido de DF; si el resultado de un bucle de comparación depende de datos desconocidos, lo conserva simbólico. `CPUID` deja sus salidas desconocidas en vez de reutilizar valores obsoletos. En x86/x86-64 aplica perfiles ABI parciales de `abi.py` para localizar parámetros, retornos y registros volátiles/preservados; en x86 también recupera los slots de pila relativos a la entrada y los argumentos apilados recientes. La selección de convención y el cálculo de aridad siguen siendo heurísticos, tal como se detalla en la sección ABI. La eliminación experimental de un retorno `EAX`/`RAX` para declarar `void` solo se aplica cuando los sitios de llamada recuperados comprueban que se sobrescribe antes de leerse; una lectura previa cuenta como retorno usado. Si hay xrefs irresolubles o flujo ambiguo, no se descarta el valor observado. En PE x64, los cuatro slots de home/shadow space (`[rsp+8]` a `[rsp+0x20]` relativos al RSP de entrada) se nombran `home_slot_1` a `home_slot_4`, no como parámetros formales; los argumentos realmente pasados en pila empiezan en `param_5`. Los `call $+5` se modelan como captura de la dirección de retorno; un `jmp` terminal a otra función ya recuperada se representa como tail-call con prototipo/argumentos desconocidos.

Para las instrucciones x86 se contrastó la semántica con `Ghidra/Processors/x86/data/languages/ia.sinc`: allí `CPUID` advierte explícitamente que los valores no son correctos y expone sus efectos, las macros `rep`/`repe`/`repne` definen el manejo de CX/ECX/RCX, y las reglas de `MOVS*`, `CMPS*`, `SCAS*`, `MOVD`, `MOVQ`, `PUNPCKLDQ`, MMX y SSE (incluidas conversiones escalares) muestran los resultados trasladados a registros/memoria. Para contrastar ABI se revisaron `Ghidra/Processors/x86/data/languages/x86win.cspec`, `x86gcc.cspec` y `x86-64-win.cspec`, que describen ubicaciones de parámetros, registros de retorno y limpieza de pila para modelos concretos. MiniGhidraPy implementa reglas Python explícitas para una parte de ese repertorio, no un traductor de SLEIGH ni un clasificador completo de prototipos.

El paralelismo con Ghidra es arquitectónico, no una copia del algoritmo: no se genera ni ejecuta p-code/SLEIGH; flags o efectos que no se puedan determinar quedan simbólicos; los parámetros de llamada se infieren solo por escrituras recientes a registros ABI o `push`; el aliasing de memoria, AVX/YMM y numerosas instrucciones/estados de máquina aún no se modelan. Aunque se añadieron los casos implícitos `MUL`/`IMUL`/`DIV`/`IDIV`, las formas y tamaños no cubiertos se conservan como aproximaciones o valores desconocidos. Los `phi` son expresiones de salida, no variables high-level de Ghidra; no se recuperan tipos, prototipos, estructuras, variables globales ni nombres equivalentes a `HighFunction`; y aún no se estructuran bucles/condiciones como C. Por ello el resultado es experimental y no promete pseudocódigo idéntico ni semántica completa para binarios arbitrarios.

La comparación estática con la sesión Ghidrathon adjunta de `gxscc.exe` sirvió para priorizar MMX, x87 y cadenas repetidas. Tras estas implementaciones, `FUN_411F60` (293 instrucciones), `FUN_412340` (527), `FUN_401910` (759) y las otras seis funciones revisadas de GXSCC generan pseudocódigo sin instrucciones marcadas `unsupported`. Al contrastar `FUN_40A150`, MiniGhidraPy ahora recupera tres parámetros de pila y un retorno `void`, coincidiendo con la aridad y el tipo de retorno mostrados por Ghidra (`void FUN_0040a150(undefined4 param_1,int param_2,int param_3)`). La tipificación Python sigue siendo uniforme y aproximada (`uint32_t`) y el cuerpo no es equivalente: mantiene bloques/gotos y expresiones `phi`. Las diez variables `phi_pending_push_*` de esa salida tienen una asignación `phi(...)` y un uso posterior como argumentos; no quedan pendientes sin resolver. La evidencia estática clasifica `FUN_40B7C0` y `FUN_40B1F0` como `__thiscall` con purgas de 28 y 8 bytes; el `this` de ECX se conserva en la firma formal, pero no aparece como argumento explícito. Así, la llamada recuperada a `FUN_40B7C0` emite siete argumentos, igual que la salida Ghidra de referencia. En el PE x64 adjunto `1tracker_x64.exe`, cuatro rutinas representativas de 165–325 instrucciones (`FUN_1400D9790`, `FUN_1400FE0E0`, `FUN_1401092B0` y `FUN_1400FA850`) tampoco dejan instrucciones marcadas `unsupported` tras cubrir conversiones y operaciones SSE. Estos resultados son evidencia estática acotada, no paridad global ni validación de equivalencia semántica. Ambos PE se inspeccionaron solo estáticamente y no se ejecutaron.

Se añadieron regresiones para plegado de constantes, parámetros de pila x86, slots home PE x64, comparaciones/ramas, uniones con `phi`, argumentos/retornos de llamadas —incluida la eliminación conservadora de un retorno no observado—, thunks `call`-next, tail-calls, `ADC`/`SBB`, multiplicación/división implícitas, x87, `CPUID`, MMX, movimientos/shifts SSE, conversión SSE escalar y `REP MOVSD` con DF. El bloque ABI cubre defaults PE/ELF, fastcall, thiscall, GCC regparm3, grupos y argumentos float Windows x64, retornos EDX:EAX/ST0/XMM0, la firma explícita de `Function` y su exportación JSON. La validación actual terminó con 73/73 pruebas, `py_compile`, Ruff en los módulos ABI/decompilador/API/pruebas y `git diff --check` correctos. El chequeo Ruff de todo `model.py`, `analysis.py`, `cli.py` y `gui.py` aún presenta avisos preexistentes fuera de este cambio; no se reformatearon masivamente esos módulos.
