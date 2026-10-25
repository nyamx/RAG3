"""Measure isolated Docling PDF conversion without touching Qdrant or job state."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--pages", type=int, default=8)
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be positive")

    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.FAST
    options.generate_picture_images = False
    options.do_picture_classification = False
    options.do_picture_description = False
    options.generate_page_images = False
    options.generate_table_images = False
    options.images_scale = 1.0
    options.queue_max_size = 8
    options.accelerator_options.device = (
        AcceleratorDevice.CUDA if args.device == "cuda" else AcceleratorDevice.CPU
    )
    options.accelerator_options.num_threads = 2
    options.artifacts_path = Path(__file__).resolve().parents[1] / "data" / "models" / "docling"
    started = time.perf_counter()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    converter.initialize_pipeline(InputFormat.PDF)
    prepared = time.perf_counter()
    result = converter.convert(args.source, page_range=(1, args.pages))
    converted = time.perf_counter()
    output = {
        "device": args.device,
        "requested_pages": args.pages,
        "status": str(result.status),
        "preparation_seconds": round(prepared - started, 3),
        "conversion_seconds": round(converted - prepared, 3),
        "total_seconds": round(converted - started, 3),
        "document_pages": len(result.document.pages),
        "text_characters": len(result.document.export_to_markdown()),
    }
    if args.device == "cuda":
        import torch

        output["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        output["torch_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
