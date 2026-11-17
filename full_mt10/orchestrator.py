"""Shared one-lane full MT10 timing schedule; no side work or pooled hosts."""
import argparse, csv, datetime, hashlib, json, os, signal, shutil, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpu-smoke',action='store_true');p.add_argument('--reserve-gib',type=int,default=45); p.add_argument('--wall-seconds',type=int,default=5300)
 a=p.parse_args()
 assert a.seed in {17711,17712,17713} and 300<=a.wall_seconds<=5300 and a.reserve_gib>=20
 assert len(os.sched_getaffinity(0))==12
 os.chdir(R)
 for h,n in [x.split('  ',1) for x in (R/'SOURCE_PINS.sha256').read_text().splitlines()]: assert sha(R/n)==h,n
 d=load(R/'design.json'); uuid=a.gpu_uuid; cpu=a.cpu_smoke
 assert cpu==(uuid=='cpu') and os.environ.get('CUDA_VISIBLE_DEVICES')==('' if cpu else uuid)
 status=R/'status.json'
 if not cpu:
  assert shutil.disk_usage(R).free>=a.reserve_gib*1024**3
  query=lambda *x:subprocess.check_output(['nvidia-smi',*x],text=True,timeout=8)
  ids=query('--query-gpu=uuid','--format=csv,noheader').splitlines()
  assert uuid in ids and not any(uuid in z for z in query('--query-compute-apps=gpu_uuid,pid','--format=csv,noheader').splitlines())
 save(status,{'verified_gpu_uuids':[] if cpu else [uuid]})
 save(R/'INVOCATION.json',dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),started_unix=time.time(),seed=a.seed,uuid=uuid,manifest=sha(R/'SOURCE_PINS.sha256'),cpu_smoke=cpu,affinity=sorted(os.sched_getaffinity(0))))
 hard=time.monotonic_ns()+(330 if cpu else a.wall_seconds)*10**9
 arms=['parent'] if cpu else d['arm_order']
 for arm in arms:
  now=time.monotonic_ns(); rem=(hard-now)/1e9
  parent_steps=1000 if cpu else d['parent_steps']; quantum=1000 if cpu else d['rollout_per_quantum']
  stages_left=len(arms)-arms.index(arm)-1
  deadline=hard-int(stages_left*(750 if arm=='parent' else 650)*10**9)
  assert deadline-now>100*10**9
  cfg=dict(phase='validation' if cpu else 'development',method=arm,run_id=arm,variant=d['campaign'],seed=a.seed,benchmark_seed=d['benchmark_seed'],insertion_seed=a.seed+500,parent_steps=parent_steps,training_steps_cap=parent_steps if arm=='parent' else d['train_steps'],grow_after_steps=d['grow_after_steps'],rollout_per_quantum=quantum,assessment_goals_per_task=1 if cpu else d['assessment_goal_count_per_task'],final_assessment_reserve_s=25 if arm=='parent' else 105,cleanup_reserve_s=20,initial_quantum_upper_s=30 if cpu else 100,author_config='configs/learner.yaml',learning_rate=.0003)
  path=R/'configs'/('effective_'+arm+'.json'); save(path,cfg)
  cmd=[sys.executable,'-B',str(R/'worker.py'),'--campaign',str(R),'--config',str(path),'--uuid',uuid,'--allocation-start-ns',str(now),'--deadline-ns',str(deadline)]
  print('launch',arm,'remaining',round(rem,2),'at',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
  with (R/(arm+'.log')).open('x') as f:
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   def stop(*_):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(15)
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     child.wait()
    sys.exit('deadline or stop; no automatic retry')
   signal.signal(signal.SIGTERM,stop)
   try:code=child.wait((deadline-time.monotonic_ns())/1e9)
   except subprocess.TimeoutExpired:stop()
  s=load(R/'runs'/arm/'state.json')
  assert code==0 and s['status']=='completed',(arm,code,s.get('failure'),s.get('training_stop_reason'))
  if not cpu:assert shutil.disk_usage(R).free>=a.reserve_gib*1024**3
 if cpu:return
 p=R/'runs'; records={}; panels={}; ds={}
 for arm in arms[1:]:
  r=p/arm;s=load(r/'state.json');ds[arm]=load(r/'late_boundary_before_growth_transport.json')
  vals=[json.loads(l) for l in (r/'evaluations.jsonl').read_text().splitlines()]
  assert [v['stream'] for v in vals]==['fork','late_boundary','final']+(['final_bypass'] if arm in ('now','later') else [])
  for v in vals:
   panel=sorted((z['task_name'],z['goal_index'],z['reset_rng_hash']) for z in v['episodes'])
   assert len(panel)==100 and panels.setdefault('fixed',panel)==panel
  records[arm]={'parent_hash':s['parent_checkpoint_sha256'],'actor_parameters':s['actor_parameters'],'physical_train_steps':s['training_steps'],'elapsed_seconds':s['elapsed_s'],'assess_steps':s['final_assessment_steps'],'panels':{v['stream']:{'steps':v['training_steps'],'success':v['success'],'per_task':dict(zip(s['task_names'],v['per_task_success']))} for v in vals}}
 assert len({v['parent_hash'] for v in records.values()})==1
 same=ds['keep']==ds['later']
 save(R/'summary.json',dict(seed=a.seed,uuid=uuid,parent_steps=d['parent_steps'],records=records,prefix_exact_until_300k=same,scope='pilot independent learner restarts (not native simulator resume), no GKP superiority claim'))
 with (R/'summary.csv').open('x') as f:
  w=csv.writer(f);w.writerow(['arm','panel','training_env_interactions','success','elapsed_seconds'])
  for k,s in records.items():
   for phase,v in s['panels'].items():w.writerow([k,phase,v['steps'],v['success'],s['elapsed_seconds']])
if __name__=='__main__':main()
