"""Generate the small RAG3 launcher icon without external image assets."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


target = Path(__file__).with_name("RAG3.ico")
size = 256
image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((8, 8, 248, 248), radius=58, fill="#196b45")
draw.rounded_rectangle((20, 20, 236, 236), radius=46, outline="#64c991", width=8)
font_path = Path("C:/Windows/Fonts/seguisb.ttf")
font = ImageFont.truetype(str(font_path), 112) if font_path.exists() else ImageFont.load_default()
bounds = draw.textbbox((0, 0), "R3", font=font)
width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
draw.text(((size - width) / 2 - bounds[0], (size - height) / 2 - bounds[1] - 6),
          "R3", fill="white", font=font)
image.save(target, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                        (64, 64), (128, 128), (256, 256)])
print(target)
