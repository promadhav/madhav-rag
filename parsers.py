"""Mixed document readers. Python documents are read as text, never executed.

The worker calls this in a short-lived process with a per-file timeout, so a
broken parser/file cannot block the rest of the corpus indefinitely.
"""
import csv
import io
import json
import os
import posixpath
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

SUPPORTED = {'.pdf','.docx','.xlsx','.csv','.txt','.log','.py','.png','.jpg','.jpeg','.tif','.tiff'}
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def xml(z, name):
    entry = z.getinfo(name)
    if entry.file_size > 128 * 1024**2:
        raise ValueError('XML entry too large for bounded parser')
    return ET.fromstring(z.read(name))


def texts(el, tag):
    return ''.join(e.text or '' for e in el.iter(tag))


def row_text(headers, row):
    return ' | '.join(f'{headers[i] if i < len(headers) and headers[i] else "column_"+str(i+1)}: {value}'
                      for i,value in enumerate(row) if value.strip())


def parse_docx(path):
    parts = []
    with zipfile.ZipFile(path) as z:
        root = xml(z,'word/document.xml')
        body = root.find(W+'body')
        context = []
        for child in body:
            if child.tag == W+'p':
                t = texts(child,W+'t').strip()
                if t:
                    parts.append({'label':'paragraph','text':t})
                    context.append(t)
            elif child.tag == W+'tbl':
                rows = [[texts(c,W+'t').strip() for c in row.findall(W+'tc')]
                        for row in child.findall(W+'tr')]
                if not rows:
                    continue
                header = rows[0]
                heading = '\n'.join(context[-2:])
                for row in rows[1:] or rows:
                    parts.append({'label':'table','text':heading+'\n'+row_text(header,row)})
        for name in z.namelist():
            if name.startswith(('word/header','word/footer')) and name.endswith('.xml'):
                t = texts(xml(z,name),W+'t').strip()
                if t:
                    parts.append({'label':name,'text':t})
    return parts


def parse_xlsx(path):
    parts = []
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [texts(si,S+'t') for si in xml(z,'xl/sharedStrings.xml')]
        rels = {e.attrib['Id']: e.attrib['Target'] for e in xml(z,'xl/_rels/workbook.xml.rels')}
        for sheet in xml(z,'xl/workbook.xml').find(S+'sheets'):
            target = rels[sheet.attrib[R+'id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            rows = []
            for row in xml(z,target).iter(S+'row'):
                cells = {}
                for cell in row.findall(S+'c'):
                    column = ''.join(c for c in cell.attrib.get('r','A1') if c.isalpha())
                    idx = 0
                    for c in column:
                        idx = idx*26 + ord(c.upper())-64
                    value = cell.find(S+'v')
                    t = value.text if value is not None and value.text else ''
                    if cell.attrib.get('t') == 's' and t:
                        t = shared[int(t)]
                    elif cell.attrib.get('t') == 'inlineStr':
                        t = texts(cell,S+'t')
                    formula = cell.find(S+'f')
                    if formula is not None and not t:
                        t = '[formula, no cached value: '+(formula.text or '')+']'
                    cells[max(idx-1,0)] = t
                if cells:
                    rows.append([cells.get(i,'') for i in range(max(cells)+1)])
            if not rows:
                continue
            # Retain early title/header rows and explicit row numbers. No formula execution.
            header_idx = next((i for i,r in enumerate(rows) if sum(bool(x) for x in r)>1),0)
            header = rows[header_idx]
            context = ' | '.join(' '.join(r) for r in rows[:header_idx])
            for n,row in enumerate(rows):
                parts.append({'label':f'sheet={sheet.attrib["name"]}, row={n+1}',
                              'text':f'Sheet: {sheet.attrib["name"]}\n{context}\n'+row_text(header,row)})
    return parts


def parse(path, temp):
    path, temp = Path(path), Path(temp)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return {'parts':[], 'images':[], 'skipped':'unknown extension'}
    # Also make the mode-000 fixture meaningful in root notebook sessions.
    if path.stat().st_mode & 0o444 == 0:
        raise PermissionError('File has no read permission bits')
    with path.open('rb') as f:
        f.read(1)
    if suffix in {'.png','.jpg','.jpeg','.tif','.tiff'}:
        return {'parts':[], 'images':[{'path':str(path),'label':'image'}]}
    if suffix == '.docx':
        return {'parts':parse_docx(path),'images':[]}
    if suffix == '.xlsx':
        return {'parts':parse_xlsx(path),'images':[]}
    if suffix == '.pdf':
        import pymupdf
        parts, images = [], []
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                return {'parts':[], 'images':[], 'skipped':'encrypted PDF'}
            for i,page in enumerate(doc):
                t = page.get_text(sort=True).strip()
                if t:
                    parts.append({'label':f'page={i+1}','text':t})
                # Read scans as well as diagram-bearing pages with embedded images.
                if len(t)<60 or page.get_images():
                    temp.mkdir(parents=True,exist_ok=True)
                    dst = temp / f'page-{i+1}.png'
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(dst)
                    images.append({'path':str(dst),'label':f'page={i+1}, vision'})
        return {'parts':parts,'images':images}
    raw = path.read_bytes()
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')):
        t = raw.decode('utf-16')
    else:
        try:
            t = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            t = raw.decode('cp1252',errors='replace')
    if suffix == '.csv':
        try:
            dialect = csv.Sniffer().sniff(t[:8192],delimiters=',;\t|')
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(t),dialect))
        parts = [{'label':f'row={i+2}','text':row_text(rows[0],r)} for i,r in enumerate(rows[1:])] if rows else []
    else:
        parts = [{'label':'text','text':t}]
    return {'parts':parts,'images':[]}


if __name__ == '__main__':
    try:
        result = parse(sys.argv[1],sys.argv[2])
    except Exception as e:
        result = {'parts':[], 'images':[], 'skipped':f'{type(e).__name__}: {e}'}
    print(json.dumps(result,ensure_ascii=False))
