import json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows={}
for mode in ('baseline','forces','baseline_z','forces_z'):
    import torch
    checkpoint=torch.load(root/mode/'best.pt',map_location='cpu',weights_only=True)
    rows[mode]={'epoch':checkpoint['epoch'],'validation':checkpoint['validation'],
                'test':json.loads((root/mode/'test_metrics.json').read_text())}
(root/'z_comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
