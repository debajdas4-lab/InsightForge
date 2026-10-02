# clean_nbsp.py
from pathlib import Path
target = Path("app/streamlit_app.py")
# 1. Read as UTF-8 (so the U+00A0 characters are decoded correctly, not as raw bytes)
text = target.read_text(encoding="utf-8")
# 2. Count before touching anything, so you know what changed
count = text.count("\u00a0")
print(f"Found {count} non-breaking space(s) (U+00A0).")
if count == 0:
    print("Nothing to fix — file is already clean.")
else:
# 3. Backup the original first, in case anything looks wrong afterward
    backup = target.with_name(target.stem + "_before_nbsp_fix.py")
    backup.write_text(text, encoding="utf-8")
    print(f"Backup saved to: {backup}")
# 4. Replace U+00A0 with a normal space (U+0020) -- safe because NBSP is
# visually identical to a regular space; this never changes meaning
# or indentation structure, only the underlying character code.
cleaned = text.replace("\u00a0", " ")
target.write_text(cleaned, encoding="utf-8")
print(f"Replaced {count} occurrence(s). File saved.")
# 5. Verify the fix actually worked and the file still parses as valid Python
import ast
ast.parse(cleaned)