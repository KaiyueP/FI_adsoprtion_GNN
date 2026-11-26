"""Summarize validation-selected checkpoints and matched pooling effects."""
import csv
import json
from pathlib import Path
import sys
import torch
root=Path(sys.argv[1])
rows=[]
for mode in ('baseline','forces','baseline_adsorbate','forces_adsorbate'):
    ckpt=torch.load(root/mode/'best.pt',map_location='cpu',weights_only=True)
    history=json.loads((root/mode/'history.json').read_text())
    test=json.loads((root/mode/'test_metrics.json').read_text())
    row=dict(mode=mode,pooling=ckpt['config']['pooling'],loss=ckpt['config']['loss'],
             best_epoch=ckpt['epoch'],completed_epochs=len(history))
    for prefix,metrics in [('validation',ckpt['validation']),('test',test)]:
        row.update({f'{prefix}_{k}':v for k,v in metrics.items()})
    rows.append(row)
by_mode={row['mode']:row for row in rows}
effects={}
for base in ('baseline','forces'):
    a,b=by_mode[base],by_mode[base+'_adsorbate']
    effects[base]={f'{split}_mae_reduction_percent':100*(a[f'{split}_mae_eV']-b[f'{split}_mae_eV'])/a[f'{split}_mae_eV']
                   if a[f'{split}_mae_eV'] else None for split in ('validation','test')}
(root/'pooling_comparison.json').write_text(json.dumps(dict(runs=rows,pooling_effect=effects),indent=2))
with (root/'pooling_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(runs=rows,pooling_effect=effects),indent=2))
