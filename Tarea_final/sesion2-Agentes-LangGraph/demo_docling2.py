from docling.document_converter import DocumentConverter

pdf_file = "articulo.pdf"

# ------------------------------------------------------------------
# Conversión
# ------------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert(pdf_file)

doc = result.document

# ------------------------------------------------------------------
# Inspección de elementos
# ------------------------------------------------------------------

print("\n")
print("=" * 80)
print("ELEMENTOS DEL DOCUMENTO")
print("=" * 80)

for i, item in enumerate(doc.iterate_items()):

    # iterate_items() devuelve (elemento, nivel)
    element, level = item

    element_type = type(element).__name__

    print(f"\n[{i}] Tipo: {element_type}")
    print(f"Nivel: {level}")

    # Intentar obtener texto asociado
    text = ""

    if hasattr(element, "text"):
        text = element.text

    elif hasattr(element, "orig"):
        text = str(element.orig)

    text = text.replace("\n", " ")

    print("Contenido:")
    print(text[:150])