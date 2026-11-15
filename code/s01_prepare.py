"""s01: both catalogues to the common 16A-wellhead frame, spatial filter, stage labels."""
import pandas as pd, numpy as np
import fz

import os
# stage times come from the 1-s injection table if present, otherwise from data/stage_times.csv (same values)
st = fz.stage_table() if os.path.exists('data/injection_2024_stages.csv') else pd.read_csv('data/stage_times.csv')
st.to_csv('results/stage_table.csv', index=False)
out = {}
for name, f in [('Niemz', fz.read_niemz), ('Zhu', fz.read_zhu)]:
    c = f(); n0 = len(c)
    c = fz.assign_stage(fz.spatial_filter(c), st)
    c.insert(0, 'id', np.arange(len(c)))
    c.to_csv(f'data/catalog_{name}.csv', index=False)
    out[name] = c
    print(f'{name}: raw {n0}, kept {len(c)}, in 16A windows {c.stage.isin(fz.STAGES_16A).sum()}, '
          f'in 16B windows {c.stage.str.startswith("16B").sum()}, t {c.t.min()/3600:.1f}-{c.t.max()/3600:.1f} h')
cnt = pd.DataFrame({k: v.stage.value_counts() for k, v in out.items()}).reindex(st.stage).fillna(0).astype(int)
print(cnt)
cnt.to_csv('results/stage_counts.csv')
