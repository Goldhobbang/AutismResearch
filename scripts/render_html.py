import markdown
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    md_text = (root / 'reports' / 'FINAL_REPORT_KO.md').read_text(encoding='utf-8')
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])
    
    template = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FINAL_REPORT_KO</title><style>body{margin:0;background:#f0f5f8;color:#173346;font:16px/1.85 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1420px;margin:auto;padding:36px;background:white}h1{font-size:2rem;line-height:1.45;color:#153a50}h2{margin-top:3rem;padding-top:1rem;border-top:2px solid #c6d9e4;color:#175b72}h3{margin-top:2rem}a{color:#08679a;overflow-wrap:anywhere}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:.91rem;margin:1.4rem 0}th,td{border:1px solid #cedce5;padding:10px;vertical-align:top;min-width:100px}th{background:#eaf3f7;text-align:left}tr:nth-child(even){background:#f8fafc}pre{background:#172c3b;color:#f1f7fa;padding:18px;overflow-x:auto;border-radius:7px;font-size:.88rem;line-height:1.6}code{font-family:Consolas,monospace}blockquote{border-left:4px solid #2e8798;margin-left:0;padding:12px 22px;background:#eef7fa}.toc{background:#edf5f8;padding:15px 24px;border-radius:9px}img{max-width:100%;height:auto}.status{font-size:.88rem;color:#546e7d}@media(max-width:700px){main{padding:18px}h1{font-size:1.5rem}}@media print{main{padding:0}body{font-size:10pt}table{display:table;font-size:8pt}th,td{min-width:0;padding:5px}pre{white-space:pre-wrap;background:#f0f0f0;color:black}.toc{display:none}h2{break-after:avoid}}</style></head><body><main><p class="status">2026-10-24 | ESP32-S3 BLE-only | WESAD 15명 실데이터 LOSO 완료 (LR AP .9048, RF .8586)</p>""" + html_body + """</main></body></html>"""
    
    (root / 'reports' / 'FINAL_REPORT_KO.html').write_text(template, encoding='utf-8')
    (root / 'FINAL_REPORT_KO.html').write_text(template, encoding='utf-8')
    print("Successfully rendered FINAL_REPORT_KO.html")

if __name__ == '__main__':
    main()
