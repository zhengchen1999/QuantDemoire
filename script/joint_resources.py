"""Record static resource limits, not achieved occupancy or bandwidth."""
import argparse
from kernel.ops import extension
from quantdemoire.experiment import write_json
from quantdemoire.precision import configure,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();configure();op=extension();rows=[]
    fields=['registers_per_thread','static_shared_bytes','dynamic_shared_bytes','epilogue_shared_bytes',
            'local_bytes_per_thread','threads_per_block','max_resident_blocks_per_sm']
    for bits in (4,8):
        for warps,access,variant in [(4,4,0),(4,8,0),(8,4,0),(8,4,1),(8,4,2),(8,4,3),(8,4,4),(8,4,5),(8,4,6),(8,4,7),(8,4,8),(8,4,9)]:
            for sparse in (False,True):
                values=op.joint_resources(bits,warps,access,variant,sparse,a.device)
                rows.append(dict(bits=bits,warps=warps,access=access,variant=variant,sparse=sparse,**dict(zip(fields,values))))
    write_json(a.output,dict(environment=metadata(),rows=rows,note=__doc__))
    for r in rows:print(r)


if __name__=='__main__':main()
