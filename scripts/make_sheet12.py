import os, glob, math
from PIL import Image, ImageDraw, ImageFont
d = "data/_staging_add12"
files = sorted(glob.glob(os.path.join(d, "*.jpg")))
cols = 3; rows = math.ceil(len(files) / cols)
cw, ch = 520, 720; pad = 12; lab = 46
W = cols * cw + (cols + 1) * pad
H = rows * (ch + lab) + (rows + 1) * pad
sheet = Image.new("RGB", (W, H), (245, 245, 245))
dr = ImageDraw.Draw(sheet)
try:
    fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
except Exception:
    fnt = ImageFont.load_default()
for i, f in enumerate(files):
    r, c = divmod(i, cols)
    x = pad + c * (cw + pad); y = pad + r * (ch + lab + pad)
    try:
        im = Image.open(f).convert("RGB"); im.thumbnail((cw, ch))
        sheet.paste(im, (x + (cw - im.width) // 2, y + lab))
    except Exception as e:
        dr.text((x + 10, y + lab + 10), f"ERR {e}", fill=(200, 0, 0), font=fnt)
    dr.rectangle([x, y, x + cw, y + ch + lab], outline=(120, 120, 120), width=2)
    nm = os.path.basename(f).replace("__", " p").replace(".jpg", "")
    dr.text((x + 6, y + 8), f"[{i+1}] {nm[:48]}", fill=(0, 0, 0), font=fnt)
out = "data/_staging_add12/_contact_sheet_add12.jpg"
sheet.save(out, quality=88)
print("SHEET", out, sheet.size, os.path.getsize(out))
