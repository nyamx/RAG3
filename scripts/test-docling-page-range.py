"""Check physical page provenance when Docling converts part of a PDF."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.haystack_native import build_parsing_pipeline, normalize_chunk
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat


def make_pdf(path: Path, page_count: int = 3) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" +
         " ".join(f"{3 + index * 2} 0 R" for index in range(page_count)) +
         f"] /Count {page_count} >>").encode(),
    ]
    for page in range(1, page_count + 1):
        contents_ref = 4 + (page - 1) * 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {3 + page_count * 2} 0 R >> >> "
            f"/Contents {contents_ref} 0 R >>".encode()
        )
        content = f"BT /F1 18 Tf 72 720 Td (UNIQUE PAGE {page} RAG3) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(content)).encode() +
                       b" >>\nstream\n" + content + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
                  f"startxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(output)


def main() -> None:
    with TemporaryDirectory(prefix="rag3-page-range-") as folder:
        source = Path(folder) / "three.pdf"
        make_pdf(source)
        pipeline = build_parsing_pipeline()
        if "--cpu" in sys.argv:
            options = pipeline.get_component("converter").converter.format_to_options[InputFormat.PDF].pipeline_options
            options.accelerator_options.device = AcceleratorDevice.CPU
            options.accelerator_options.num_threads = 2
        pipeline.get_component("converter").convert_kwargs = {"page_range": (2, 3)}
        metadata = {"source_sha256": "a" * 64, "document_id": "a" * 64,
                    "source_uri": str(source), "title": "page range"}
        started = time.monotonic()
        result = pipeline.run({"converter": {"sources": [str(source)], "meta": metadata}})
        chunks = [normalize_chunk(doc, index) for index, doc in
                  enumerate(result["splitter"]["documents"])]
        assert chunks
        text = "\n".join(chunk.content for chunk in chunks)
        assert "UNIQUE PAGE 1" not in text
        assert "UNIQUE PAGE 2" in text and "UNIQUE PAGE 3" in text
        assert all(chunk.meta["page_start"] in (2, 3) for chunk in chunks)
        print("PAGE_RANGE_PROBE_OK", [(c.meta["page_start"], c.meta["page_end"])
                                      for c in chunks], "seconds", round(time.monotonic()-started, 1))


if __name__ == "__main__":
    main()
