"""Optional Tkinter front-end; the static-analysis core remains headless."""

from __future__ import annotations

from pathlib import Path

from .image import BinaryImage, LoadError
from .project import ProjectState


def launch_gui(console) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk
    except ImportError as exc:  # pragma: no cover - platform-specific
        raise RuntimeError("Tkinter no está disponible en esta instalación de Python.") from exc

    root = tk.Tk()
    root.title(f"MiniGhidraPy — {console.image.path.name}")
    root.geometry("1220x780")
    root.minsize(900, 560)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", rowheight=23)
    style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))

    status = tk.StringVar(value="Análisis estático x86/x86-64")
    filter_text = tk.StringVar()
    address_text = tk.StringVar()
    current_function = {"value": None}
    trees = {}
    text_views = {}

    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=False)
    file_menu.add_command(label="Abrir binario…", command=lambda: open_binary())
    file_menu.add_separator()
    file_menu.add_command(label="Guardar proyecto", command=lambda: save_project())
    file_menu.add_command(label="Exportar informe JSON…", command=lambda: export_json())
    file_menu.add_command(label="Exportar ensamblador…", command=lambda: export_asm())
    file_menu.add_separator()
    file_menu.add_command(label="Salir", command=root.destroy)
    menubar.add_cascade(label="Archivo", menu=file_menu)

    edit_menu = tk.Menu(menubar, tearoff=False)
    edit_menu.add_command(label="Renombrar selección…", command=lambda: rename_selected())
    edit_menu.add_command(label="Añadir comentario…", command=lambda: comment_address())
    menubar.add_cascade(label="Editar", menu=edit_menu)
    root.configure(menu=menubar)

    toolbar = ttk.Frame(root, padding=(8, 8, 8, 4))
    toolbar.pack(fill="x")
    ttk.Label(toolbar, text="Buscar").pack(side="left")
    search_entry = ttk.Entry(toolbar, textvariable=filter_text, width=32)
    search_entry.pack(side="left", padx=(5, 12))
    ttk.Label(toolbar, text="Ir a dirección/símbolo").pack(side="left")
    address_entry = ttk.Entry(toolbar, textvariable=address_text, width=20)
    address_entry.pack(side="left", padx=5)
    ttk.Button(toolbar, text="Ir", command=lambda: go_to_address()).pack(side="left", padx=(0, 6))
    ttk.Button(toolbar, text="Guardar", command=lambda: save_project()).pack(side="left")

    panes = ttk.Panedwindow(root, orient="horizontal")
    panes.pack(fill="both", expand=True, padx=8, pady=4)
    left = ttk.Frame(panes, width=380)
    right = ttk.Frame(panes)
    panes.add(left, weight=1)
    panes.add(right, weight=3)

    left_tabs = ttk.Notebook(left)
    left_tabs.pack(fill="both", expand=True)

    def create_tree(tab_name, columns, headings, widths):
        frame = ttk.Frame(left_tabs)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="w")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        left_tabs.add(frame, text=tab_name)
        trees[tab_name] = tree
        return tree

    function_tree = create_tree("Funciones", ("address", "name", "size"), ("Dirección", "Nombre", "Bytes"), (105, 190, 55))
    string_tree = create_tree("Cadenas", ("address", "text", "encoding"), ("Dirección", "Texto", "Codificación"), (105, 205, 75))
    import_tree = create_tree("Imports", ("address", "name", "library"), ("Dirección", "Importación", "Biblioteca"), (105, 185, 95))
    export_tree = create_tree("Exports", ("address", "name"), ("Dirección", "Exportación"), (105, 250))
    section_tree = create_tree("Secciones", ("address", "name", "flags"), ("Dirección", "Sección", "R/W/X"), (105, 180, 45))

    right_tabs = ttk.Notebook(right)
    right_tabs.pack(fill="both", expand=True)

    def create_text_tab(name):
        frame = ttk.Frame(right_tabs)
        widget = tk.Text(frame, wrap="none", font=("TkFixedFont", 10), undo=False)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=widget.xview)
        widget.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set, state="disabled")
        widget.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        right_tabs.add(frame, text=name)
        text_views[name] = widget
        return widget

    listing_view = create_text_tab("Desensamblado")
    pseudo_view = create_text_tab("Pseudocódigo")
    graph_view = create_text_tab("Grafo CFG (DOT)")
    xrefs_view = create_text_tab("Referencias")
    summary_view = create_text_tab("Resumen")

    def set_text(widget, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def write_summary() -> None:
        info = console.analyzer.summary()
        lines = [
            f"Archivo: {info['file']}",
            f"Formato: {info['format']}",
            f"Arquitectura: {info['architecture']} ({info['bits']} bits)",
            f"Base: {console.image.image_base:#x}",
            f"Punto de entrada: {hex(info['entrypoint']) if info['entrypoint'] is not None else 'no disponible'}",
            f"Secciones: {info['sections']} ({info['executable_sections']} ejecutables)",
            f"Instrucciones decodificadas: {info['instructions']}",
            f"Funciones recuperadas: {info['functions']}",
            f"Thunks recuperados: {info['thunks']}",
            f"Inicios hallados por patrones x86: {info['pattern_functions']}",
            f"Funciones no-return: {info['no_return_functions']}",
            f"Imports no-return conocidos: {info['no_return_imports']}",
            f"Cadenas: {info['strings']}",
            f"Importaciones: {info['imports']}",
            f"Exportaciones: {info['exports']}",
            f"Referencias cruzadas: {info['xrefs']}",
            f"SHA-256: {info['sha256']}",
            "",
            "El programa no se ejecuta. El pseudocódigo es aproximado y no equivale a un decompilador.",
        ]
        if console.image.libraries:
            lines.extend(["", "Bibliotecas:", *[f"  {name}" for name in console.image.libraries]])
        set_text(summary_view, "\n".join(lines))

    def populate_lists(query: str = "") -> None:
        folded = query.casefold()
        for tree in trees.values():
            for item in tree.get_children():
                tree.delete(item)
        for function in sorted(console.analyzer.functions.values(), key=lambda item: item.address):
            if folded and folded not in function.name.casefold() and folded not in f"{function.address:x}":
                continue
            iid = f"f_{function.address:x}"
            function_tree.insert("", "end", iid=iid, values=(f"{function.address:#x}", function.name, function.size))
        for item in console.analyzer.strings:
            if folded and folded not in item.text.casefold() and folded not in f"{item.address:x}":
                continue
            iid = f"s_{item.address:x}_{len(string_tree.get_children())}"
            text = item.text.encode("unicode_escape").decode("ascii")
            string_tree.insert("", "end", iid=iid, values=(f"{item.address:#x}", text, item.encoding))
        for symbol in console.image.imports:
            if folded and folded not in symbol.name.casefold():
                continue
            iid = f"i_{symbol.address:x}_{len(import_tree.get_children())}"
            import_tree.insert("", "end", iid=iid, values=(f"{symbol.address:#x}" if symbol.address else "—", symbol.name, symbol.library or ""))
        for symbol in console.image.exports:
            if folded and folded not in symbol.name.casefold():
                continue
            iid = f"e_{symbol.address:x}_{len(export_tree.get_children())}"
            export_tree.insert("", "end", iid=iid, values=(f"{symbol.address:#x}", symbol.name))
        for section in console.image.sections:
            if folded and folded not in section.name.casefold():
                continue
            flags = "".join(letter if enabled else "-" for letter, enabled in (("R", section.readable), ("W", section.writable), ("X", section.executable)))
            iid = f"sec_{section.start:x}_{len(section_tree.get_children())}"
            section_tree.insert("", "end", iid=iid, values=(f"{section.start:#x}", section.name, flags))

    def display_function(function) -> None:
        current_function["value"] = function
        set_text(listing_view, console.analyzer.format_function(function))
        set_text(pseudo_view, console.analyzer.pseudocode(function))
        set_text(graph_view, console.analyzer.cfg_dot(function))
        refs = []
        for instruction in function.instructions:
            for xref in console.analyzer.xrefs_from(instruction.address):
                target_name = console.analyzer.name_at(xref.target)
                refs.append(f"{xref.source:#x} -> {xref.target:#x}  {xref.kind}" + (f" ({target_name})" if target_name else ""))
        set_text(xrefs_view, "\n".join(refs) if refs else "No hay referencias salientes desde la primera instrucción.")
        status.set(f"{function.name} @ {function.address:#x} — {len(function.instructions)} instrucciones")
        right_tabs.select(0)

    def show_address(address: int) -> None:
        try:
            function = console.analyzer.function_at(address)
        except ValueError:
            function = None
        if function is not None:
            display_function(function)
            return
        instructions = console.analyzer.iter_instructions(address, 160)
        if not instructions:
            raise ValueError(f"No hay instrucciones decodificadas desde {address:#x}.")
        current_function["value"] = None
        set_text(listing_view, "\n".join(console.analyzer.format_instruction(item) for item in instructions))
        right_tabs.select(0)
        status.set(f"Desensamblado desde {address:#x}")

    def on_function_selected(_event=None) -> None:
        selection = function_tree.selection()
        if not selection:
            return
        values = function_tree.item(selection[0], "values")
        try:
            function = console.analyzer.functions[int(values[0], 0)]
        except (KeyError, ValueError):
            return
        display_function(function)

    def on_string_selected(_event=None) -> None:
        selection = string_tree.selection()
        if not selection:
            return
        values = string_tree.item(selection[0], "values")
        try:
            address = int(values[0], 0)
        except ValueError:
            return
        references = console.analyzer.xrefs_to(address)
        lines = [f"Referencias a la cadena en {address:#x}:"]
        if not references:
            lines.append("No se detectaron referencias directas a esta dirección.")
        for xref in references:
            lines.append(f"{xref.source:#x} -> {xref.target:#x}  {xref.kind}")
        set_text(xrefs_view, "\n".join(lines))
        right_tabs.select(3)
        status.set(f"Cadena en {address:#x}")

    def on_symbol_selected(event=None) -> None:
        tree = event.widget if event is not None else None
        if tree is None or not tree.selection():
            return
        values = tree.item(tree.selection()[0], "values")
        if not values:
            return
        try:
            address = int(values[0], 0)
            show_address(address)
        except (ValueError, TypeError):
            pass

    function_tree.bind("<<TreeviewSelect>>", on_function_selected)
    string_tree.bind("<<TreeviewSelect>>", on_string_selected)
    import_tree.bind("<Double-1>", on_symbol_selected)
    export_tree.bind("<Double-1>", on_symbol_selected)
    section_tree.bind("<Double-1>", lambda _event: show_selected_section())

    def show_selected_section() -> None:
        selection = section_tree.selection()
        if not selection:
            return
        values = section_tree.item(selection[0], "values")
        try:
            address = int(values[0], 0)
            show_address(address)
        except (ValueError, TypeError) as exc:
            status.set(str(exc))

    def on_filter_change(*_args) -> None:
        populate_lists(filter_text.get())

    filter_text.trace_add("write", on_filter_change)

    def go_to_address() -> None:
        try:
            address = console.analyzer.resolve_address(address_text.get())
            show_address(address)
        except ValueError as exc:
            messagebox.showerror("Dirección no válida", str(exc), parent=root)

    address_entry.bind("<Return>", lambda _event: go_to_address())

    def selected_address() -> Optional[int]:
        tree = left_tabs.nametowidget(left_tabs.select())
        if not isinstance(tree, ttk.Frame):
            return None
        # Find which tree is currently shown in the selected notebook frame.
        for name, candidate in trees.items():
            if str(candidate.master) == str(tree) and candidate.selection():
                values = candidate.item(candidate.selection()[0], "values")
                if values:
                    try:
                        return int(values[0], 0)
                    except (ValueError, TypeError):
                        return None
        return None

    def rename_selected() -> None:
        address = selected_address()
        if address is None:
            function = current_function["value"]
            address = function.address if function is not None else None
        if address is None:
            messagebox.showinfo("Renombrar", "Selecciona una función o indica una dirección.", parent=root)
            return
        old_name = console.analyzer.name_at(address) or f"FUN_{address:X}"
        new_name = simpledialog.askstring("Renombrar etiqueta", "Nuevo nombre:", initialvalue=old_name, parent=root)
        if not new_name:
            return
        console.state.labels[address] = new_name
        console.analyzer._name_by_address[address] = new_name
        if address in console.analyzer.functions:
            console.analyzer.functions[address].name = new_name
        populate_lists(filter_text.get())
        status.set(f"Etiqueta {address:#x} = {new_name} (pendiente de guardar)")

    def comment_address() -> None:
        address = selected_address()
        if address is None:
            function = current_function["value"]
            address = function.address if function is not None else None
        if address is None and address_text.get():
            try:
                address = console.analyzer.resolve_address(address_text.get())
            except ValueError:
                address = None
        if address is None:
            messagebox.showinfo("Comentario", "Selecciona una dirección o escríbela en la barra superior.", parent=root)
            return
        old = console.state.comments.get(address, "")
        comment = simpledialog.askstring("Comentario", f"Comentario para {address:#x}:", initialvalue=old, parent=root)
        if comment is None:
            return
        console.state.comments[address] = comment
        function = current_function["value"]
        if function is not None:
            display_function(function)
        status.set(f"Comentario en {address:#x} (pendiente de guardar)")

    def save_project() -> None:
        try:
            destination = console.state.save(console.project_path, console.image)
            status.set(f"Proyecto guardado: {destination}")
        except OSError as exc:
            messagebox.showerror("Error al guardar", str(exc), parent=root)

    def export_json() -> None:
        destination = filedialog.asksaveasfilename(
            parent=root, title="Exportar informe", defaultextension=".json", filetypes=[("JSON", "*.json"), ("Todos", "*")]
        )
        if not destination:
            return
        try:
            console._export_json([destination])
        except (ValueError, OSError) as exc:
            messagebox.showerror("Error al exportar", str(exc), parent=root)

    def export_asm() -> None:
        destination = filedialog.asksaveasfilename(
            parent=root, title="Exportar desensamblado", defaultextension=".asm", filetypes=[("Ensamblador", "*.asm"), ("Todos", "*")]
        )
        if not destination:
            return
        try:
            console._export_asm([destination])
        except (ValueError, OSError) as exc:
            messagebox.showerror("Error al exportar", str(exc), parent=root)

    def open_binary() -> None:
        selected = filedialog.askopenfilename(parent=root, title="Abrir ejecutable")
        if not selected:
            return
        try:
            image = BinaryImage.load(selected)
        except LoadError as exc:
            messagebox.showerror("No se pudo abrir", str(exc), parent=root)
            return
        path = ProjectState.default_path(image.path)
        try:
            state = ProjectState.load(path, image) if path.exists() else ProjectState()
        except ValueError as exc:
            messagebox.showwarning("Proyecto ignorado", str(exc), parent=root)
            state = ProjectState()
        try:
            from .cli import Console

            new_console = Console(image, state, path)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Error de análisis", str(exc), parent=root)
            return
        console.__dict__.update(new_console.__dict__)
        root.title(f"MiniGhidraPy — {console.image.path.name}")
        populate_lists(filter_text.get())
        write_summary()
        status.set(f"Abierto {console.image.path}")

    populate_lists()
    write_summary()
    if console.image.entrypoint is not None:
        try:
            show_address(console.image.entrypoint)
        except ValueError:
            pass

    ttk.Label(root, textvariable=status, anchor="w", relief="sunken", padding=(6, 3)).pack(fill="x", side="bottom")
    root.mainloop()
