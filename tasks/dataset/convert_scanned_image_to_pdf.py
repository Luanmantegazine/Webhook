"""Convert a scanned RVL-CDIP image into the PDF input expected by Hydra."""

from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image, ImageSequence

from core.task import task


@task(
    outputs={
        "pdf_path": {
            "type": "str",
            "description": "Path to the generated PDF",
        }
    },
    display_name="Convert Scanned Image To PDF",
    description="Wrap a TIFF, PNG, or JPEG scan as a PDF so it can enter the existing Hydra document workflow",
    category="document",
    parameters={
        "image_path": {
            "type": "str",
            "required": True,
            "description": "Path to a TIFF, PNG, or JPEG scan",
        },
        "output_directory": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Directory for the generated PDF; a temporary directory is used when empty",
        },
        "dpi": {
            "type": "int",
            "required": False,
            "default": 300,
            "description": "Resolution metadata written into the generated PDF",
        },
    },
)
def convert_scanned_image_to_pdf(
    image_path: str,
    output_directory: str = "",
    dpi: int = 300,
) -> tuple:
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scanned image not found: {source}")
    if source.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise ValueError("image_path must be a TIFF, PNG, or JPEG file")

    if output_directory:
        destination_dir = Path(output_directory).expanduser().resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
    else:
        destination_dir = Path(tempfile.mkdtemp(prefix="hydra-rvl-"))
    destination = destination_dir / f"{source.stem}.pdf"

    with Image.open(source) as image:
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"No image frames found in {source}")
    frames[0].save(
        destination,
        format="PDF",
        save_all=len(frames) > 1,
        append_images=frames[1:],
        resolution=max(72, int(dpi)),
    )
    return (str(destination),)

