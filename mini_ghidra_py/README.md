# MiniGhidraPy

Una herramienta pequeña, escrita en Python, para explorar estáticamente ejecutables **x86 de 32 bits y x86-64**. Está aislada del código Java original de Ghidra y sirve como prototipo/ayuda de aprendizaje; no es un reemplazo de Ghidra.

## Qué incluye

- Carga de ejecutables **ELF y PE** mediante LIEF y análisis de ficheros crudos con arquitectura/base indicadas. Mach-O x86 puede funcionar según la versión de LIEF, pero todavía no es una ruta de prueba garantizada.
- Detección automática de x86/x86-64 para ELF/PE; desensamblado Intel con Capstone.
- Resumen y secciones, símbolos, importaciones/exportaciones y cadenas ASCII/UTF-16LE.
- Descubrimiento heurístico de funciones (punto de entrada, símbolos, destinos de llamadas, firmas x86, inicios Mach-O `LC_FUNCTION_STARTS`, rangos `.pdata`/`UNWIND_INFO` de PE x64 y callbacks TLS de PE), thunks de salto directos/PLT, marcado de funciones conocidas `no-return` e inferencia conservadora por evidencias de llamadas/wrappers.
- Estimación de la purga de pila a partir de `ret`/`ret imm16` cuando los retornos recuperados coinciden; los perfiles ABI x86/x86-64 separan esta evidencia de la inferencia heurística de convenciones y parámetros.
- Referencias cruzadas a llamadas/saltos, memoria y operandos escalares; detección conservadora de tablas de punteros a código y destinos CFG de saltos indirectos; pseudocódigo semántico experimental con perfiles ABI x86/x86-64, firmas parciales, propagación simbólica de registros/pila, expresiones aritméticas simplificadas, condiciones y uniones CFG mediante `phi(...)`.
- Búsqueda de texto/bytes, etiquetas y comentarios guardados en un proyecto JSON, y parcheo seguro a un **archivo de salida** (el original no se modifica).
- Consola interactiva y una interfaz gráfica Tkinter opcional para recorrer funciones, cadenas, imports/exports, desensamblado, pseudocódigo y bloques.

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

Esta iteración incorpora detección conservadora de prólogos x86/x86-64, inicios Mach-O `LC_FUNCTION_STARTS`, PE x64 `.pdata`/`UNWIND_INFO` y callbacks TLS, referencias escalares, tablas absolutas de punteros, resolución de saltos de tabla, catálogos `no-return`, purga de pila desde `ret`, thunks PLT, perfiles parciales de convenciones/firmas x86/x86-64 y el primer prototipo de pseudocódigo semántico. Las fuentes revisadas, las reglas trasladadas y las diferencias con el motor original están documentadas en [`PORTING_NOTES.md`](PORTING_NOTES.md).

## Límites conocidos

Esto es una **primera versión funcional**, no tiene paridad con Ghidra. El pseudocódigo ya aplica un subconjunto de decompilación semántica (propaga registros y slots de pila, simplifica operaciones enteras, expresa condiciones de flags y usa `phi(...)` en uniones), pero conserva `goto`, no estructura todos los bucles/condiciones y deja comentarios para instrucciones no soportadas. Aún no incluye la traducción SLEIGH/p-code, inferencia completa de tipos/prototipos, análisis de variables globales, optimización general ni el motor de impresión de Ghidra; por tanto no promete pseudocódigo idéntico ni una semántica completa para cualquier instrucción. El descubrimiento de funciones y referencias sigue siendo heurístico; tampoco hay depurador, emulación ni análisis de ejecución dinámica, base de datos de tipos, scripting/plugin API, colaboración ni soporte de otras arquitecturas. El alcance actual del desensamblador está limitado a x86 de 32 bits y x86-64. La importación de formatos depende de las capacidades de LIEF instalado.

## Pruebas

```bash
python -m unittest discover -s tests -v
```
