"""Run sweep configs on Modal; each call returns a tar.gz of the run's logs (no checkpoints).

modal run --detach modal_job.py --specs "fixed:2:2,grow:1:2" --outdir ../modal_results
spec = method:depth:seed:width[:growM-growM]  e.g. grow:1:2:400:1-2
"""
import io, subprocess, tarfile, os
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('libegl1', 'libgl1', 'libglib2.0-0')
         .pip_install_from_requirements(str(HERE / 'resolved.lock'))
         .add_local_dir(str(HERE), '/kit', ignore=['runs', '__pycache__', '*.pt', 'analysis']))
app = modal.App('depth-sweep', image=image)
vol = modal.Volume.from_name('depth-sweep-results', create_if_missing=True)


@app.function(gpu='T4', cpu=12, memory=16384, timeout=4 * 3600, volumes={'/results': vol})
def run(spec: str):
    method, depth, seed, width, *grow = spec.split(':')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()[0]
    name = f'{method}_w{width}_d{depth}' + (f'_g{grow[0]}' if grow else '') + f'_s{seed}'
    out = f'/tmp/{name}'
    cmd = ['python', '/kit/run_one.py', '--out', out, '--method', method, '--depth', depth, '--seed', seed, '--uuid', uuid, '--width', width]
    if method == 'grow':
        cmd += ['--grow-at'] + [str(int(float(x) * 1e6)) for x in (grow[0] if grow else '1-2').split('-')]
    code = subprocess.call(cmd)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        tar.add(out, arcname=name, filter=lambda t: None if t.name.endswith('.pt') or '/reference' in t.name else t)
    Path(f'/results/{name}.tgz').write_bytes(buf.getvalue()); vol.commit()
    return name, code


@app.local_entrypoint()
def main(specs: str):
    for name, code in run.map(specs.split(','), return_exceptions=True, order_outputs=False):
        print('done', name, code)
