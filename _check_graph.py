import nbformat
nb = nbformat.read(r'c:/Users/diogo/work_code/ds-projects-portfolio/notebooks/graph_fraud_ring_detection.ipynb', as_version=4)
targets = ['build-graph', 'largest-components', 'communities', 'n2v', 'lgbm-train', 'lgbm-pak', 'sage-train', 'sage-pak', 'lb-align', 'link', 'ring-score', 'ring-recall', 'inference-parity', 'model-card']
buf = []
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    cid = c.get('id', '')
    if not any(k in cid for k in targets):
        continue
    buf.append('=== ' + cid + ' ===')
    for o in c.outputs:
        if o.output_type == 'stream':
            buf.append(o.text.rstrip()[:2500])
        elif o.output_type == 'execute_result':
            buf.append(o.get('data', {}).get('text/plain', '')[:2500])
    buf.append('')
out = '\n'.join(buf)
out_ascii = out.encode('ascii', errors='replace').decode('ascii')
with open(r'c:/Users/diogo/work_code/ds-projects-portfolio/_graph_outputs.txt', 'w', encoding='utf-8') as f:
    f.write(out_ascii)
print('wrote', len(out_ascii), 'chars')
