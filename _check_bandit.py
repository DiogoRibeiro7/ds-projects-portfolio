import nbformat
nb = nbformat.read(r'c:/Users/diogo/work_code/ds-projects-portfolio/notebooks/contextual_bandits_offline_evaluation.ipynb', as_version=4)
targets = ['load-obd', 'split', 'ctr-per-item', 'fit-policies', 'verify-dists', 'dm-qhat', 'ground-truth', 'ope-run', 'ope-mse-summary', 'bootstrap', 'hygiene-latency', 'hygiene-persist', 'hygiene-drift', 'model-card']
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
with open(r'c:/Users/diogo/work_code/ds-projects-portfolio/_bandit_outputs.txt', 'w', encoding='utf-8') as f:
    f.write(out_ascii)
print('wrote', len(out_ascii), 'chars')
