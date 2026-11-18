"""Convert a Markdown report to PDF (Markdown -> HTML -> headless Edge/Chrome).

    python scripts/md_to_pdf.py PHASE1_REPORT.md
"""
import os
import shutil
import subprocess
import sys

import markdown

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#1b1f24;max-width:none;margin:0}
h1{font-size:19pt;border-bottom:2px solid #d81b60;padding-bottom:4px;margin-top:0}
h2{font-size:13.5pt;color:#ad1457;margin-top:16px;border-bottom:1px solid #ddd;padding-bottom:2px}
table{border-collapse:collapse;margin:8px 0;font-size:9pt;width:100%}
th,td{border:1px solid #c9ced6;padding:3px 6px;text-align:left}
th{background:#f3f4f6}
code{background:#f3f4f6;padding:0 3px;border-radius:3px;font-size:9pt}
img{max-width:100%;margin:6px 0}
hr{border:0;border-top:1px solid #ddd}
@page{size:A4;margin:14mm}
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    shutil.which("chromium") or "", shutil.which("google-chrome") or "",
]


def main(md_path):
    md_path = os.path.abspath(md_path)
    base = os.path.splitext(md_path)[0]
    html = markdown.markdown(open(md_path, encoding="utf-8").read(), extensions=["tables", "fenced_code"])
    html_path = base + ".html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
    exe = next(b for b in BROWSERS if b and os.path.exists(b))
    pdf_path = base + ".pdf"
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    os.remove(html_path)
    print("written", pdf_path)


if __name__ == "__main__":
    main(sys.argv[1])
