import os,sys,time,subprocess,signal,shutil,csv,json
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
uuid=os.environ['CUDA_VISIBLE_DEVICES']  # local UUID supplied by launch.sh
assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
assert len(os.sched_getaffinity(0))==12
assert shutil.disk_usage(root).free>=45*1024**3
live=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True,timeout=8)
assert not any(uuid in line for line in live.splitlines())
try:
 f=os.open('INVOCATION.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(f,'w') as out:json.dump({'pid':os.getpid(),'started':time.time(),'gpu':uuid},out)
except FileExistsError:sys.exit('Existing invocation')
start=time.monotonic_ns(); hard=start+3000*10**9
for arm in ('parent','keep','identity','ordinary'):
 now=time.monotonic_ns()
 if now>=hard:sys.exit('Timeout before arm')
 arm_seconds=1400 if arm=='parent' else 600
 deadline=min(hard,now+arm_seconds*10**9)
 cmd=[sys.executable,'-B',str(root/'worker.py'),'--campaign',str(root),'--config','configs/'+arm+'.json','--uuid',uuid,'--allocation-start-ns',str(now),'--deadline-ns',str(deadline)]
 with (root/(arm+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  def stop(*_):
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:child.wait(15)
   except subprocess.TimeoutExpired:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
   sys.exit('Deadline/stop')
  signal.signal(signal.SIGTERM,stop)
  try: code=child.wait((deadline-time.monotonic_ns())/1e9)
  except subprocess.TimeoutExpired:stop()
 state=json.loads((root/'runs'/arm/'state.json').read_text())
 assert code==0 and state['status']=='completed', (arm,code,state.get('failure'))
 assert shutil.disk_usage(root).free>=45*1024**3
arms=('keep','identity','ordinary')
summary={}
panels={}
for arm in arms:
 state=json.loads((root/'runs'/arm/'state.json').read_text())
 vals=[json.loads(i) for i in (root/'runs'/arm/'evaluations.jsonl').read_text().splitlines()]
 assert [v['stream'] for v in vals]==['immediate','final']
 for v in vals:
  key=v['stream']; panel=sorted((row['task_name'],row['goal_index'],row['reset_rng_hash']) for row in v['episodes'])
  assert panels.setdefault(key,panel)==panel
 summary[arm]={'kl':state['probe_kl_parent_to_arm_mean'],'sha':state['parent_checkpoint_sha256'],'success0':vals[0]['success'],'success300k':vals[1]['success'],'steps':state['training_steps']}
assert len({x['sha'] for x in summary.values()})==1
with open('summary.json','x') as out:json.dump(summary,out,indent=2)
with open('summary.csv','x') as out:
 writer=csv.writer(out);writer.writerow(['arm','kl_parent_to_arm','success_at_0','success_at_299997','steps'])
 for arm,s in summary.items():writer.writerow([arm,s['kl'],s['success0'],s['success300k'],s['steps']])
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390" viewBox="0 0 640 390"><rect width="640" height="390" fill="#fff"/><g font-family="sans-serif" fill="#16233a"><text x="46" y="30" font-weight="bold">Fresh three-task pilot: held-out goal success (one seed)</text>']
for y in range(0,101,25):
 py=310-y*2.45;svg.append(f'<line x1="80" y1="{py}" x2="500" y2="{py}" stroke="#ddd"/><text x="40" y="{py+5}">{y}%</text>')
for i,(arm,s) in enumerate(summary.items()):
 c=('#3264ba','#15876e','#ad5920')[i];x=105+9*i;z=485+9*i;v=310-245*s['success0'];w=310-245*s['success300k']
 svg.append(f'<path d="M{x} {v} L{z} {w}" stroke="{c}" stroke-width="3" fill="none"/><circle cx="{x}" cy="{v}" r="4" fill="{c}"/><circle cx="{z}" cy="{w}" r="4" fill="{c}"/><text x="{98+i*142}" y="365" fill="{c}">{arm}</text>')
svg.append('<text x="79" y="330">Immediately after insertion</text><text x="395" y="330">After 299,997 steps</text></g></svg>')
with open('summary.svg','x') as out:out.write(''.join(svg))
