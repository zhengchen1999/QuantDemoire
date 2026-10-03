"""Report every layer and aggregate branch cost, without dropping failures."""
import argparse
import json
from pathlib import Path
import numpy as np
from quantdemoire.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();summary=[]
    for path in a.input:
        result=json.loads(Path(path).read_text())
        for bits in sorted({r['bits'] for r in result['rows']}):
            all_rows=[r for r in result['rows'] if r['bits']==bits]
            actual=[r for r in all_rows if r['weight_source']=='deployment']
            rows=actual or all_rows
            totals={key:sum(r[key]['median_ms'] for r in rows) for key in ('main','mixed','without_sparse','fp16')}
            additional=totals['mixed']-totals['without_sparse']
            summary.append(dict(input=path,bits=bits,numerical_test_count=len(all_rows),
                numerical_pass_count=sum(r['numerical_pass'] for r in all_rows),
                aggregated_layers=len(rows),aggregate_weight_source='deployment' if actual else 'synthetic',
                totals_ms=totals,incremental_branch_ms=additional,
                aggregate_branch_vs_integer_core=additional/totals['main'],
                aggregate_branch_vs_main_pipeline=additional/totals['without_sparse'],
                per_layer_30pct_integer_core_pass=sum(r['branch_30pct_pass'] for r in rows),
                per_layer_30pct_pipeline_pass=sum(r['branch_ratio_vs_main_pipeline']<=.3 for r in rows),
                worst_layer_integer_core_ratio=max(r['incremental_branch_ratio'] for r in rows),
                median_layer_integer_core_ratio=float(np.median([r['incremental_branch_ratio'] for r in rows])),
                failed_layers=[r['layer'] for r in rows if not r['branch_30pct_pass']]))
    write_json(a.output,summary)
    for row in summary:print({k:v for k,v in row.items() if k!='failed_layers'})


if __name__=='__main__':main()
