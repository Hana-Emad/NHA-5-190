
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]

ERRORS_FILE = ROOT / "artifacts/m2/clip_zero_shot/error_analysis.csv"
MANIFEST_FILE = ROOT / "artifacts/m1/garment_manifest.csv"
OUTPUT_FILE = ROOT / "artifacts/m2/clip_zero_shot/error_contact_sheet.jpg"

with ERRORS_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    errors = list(csv.DictReader(file))

with MANIFEST_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    manifest = list(csv.DictReader(file))

manifest_by_id = {row["item_id"]: row for row in manifest}

columns = 3
cell_width = 380
cell_height = 420
rows = (len(errors) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)
draw = ImageDraw.Draw(sheet)

for index, error in enumerate(errors):
    item = manifest_by_id.get(error["item_id"])

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    draw.rectangle(
        [x + 5, y + 5, x + cell_width - 5, y + cell_height - 5],
        outline="gray",
        width=2
    )

    if not item:
        draw.text((x + 15, y + 15), "Manifest record not found", fill="red")
        continue

    image_path = item.get("image_path", "").strip()

    if image_path:
        image_path = ROOT / image_path.replace("\\", "/")

    if image_path and image_path.is_file():
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_width - 30, 280))

        image_x = x + (cell_width - image.width) // 2
        sheet.paste(image, (image_x, y + 12))
    else:
        draw.text((x + 15, y + 20), "Image not found", fill="red")

    details = [
        error["item_id"],
        f"True: {error['true_category']}",
        f"Predicted: {error['predicted_category']}",
        f"Mapping: {error['mapping_status']}",
        f"Crop: {item.get('crop_method', 'unknown')}",
    ]

    text_y = y + 305
    for line in details:
        draw.text((x + 15, text_y), line, fill="black")
        text_y += 21

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_FILE, quality=90)

print(f"Images inspected: {len(errors)}")
print(f"Contact sheet saved to: {OUTPUT_FILE}")
