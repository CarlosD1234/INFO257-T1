from docling.document_converter import DocumentConverter

pdf_file = "articulo.pdf"

# Convertir PDF
converter = DocumentConverter()
result = converter.convert(pdf_file)

doc = result.document

# Exportar a Markdown
markdown = doc.export_to_markdown()

print("=" * 80)
print("DOCUMENTO EN MARKDOWN")
print("=" * 80)

print(markdown[:5000])


# Mostrar solamente los encabezados
for line in markdown.split("\n"):

    if line.startswith("#"):
        print(line)