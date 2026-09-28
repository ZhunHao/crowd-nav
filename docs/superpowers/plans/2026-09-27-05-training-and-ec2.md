# 05 — Prediction-aware retraining and AWS CLI EC2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate modern policies on the selected marine environment with reproducible, budgeted EC2 execution.

**Architecture:** First expose the same simulator as a Gymnasium step environment. Train PPO and a constraint-penalty ablation with held-out evaluation, then use a measured CPU/GPU pilot to select EC2 capacity and save resumable artifacts.

**Tech Stack:** Modern Python/PyTorch/Gymnasium/SB3 locks; AWS CLI v2, EC2, SSM and private S3 artifacts

**Spec:** [Revised design](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` and Git-ignored.
- Use `.venv-legacy` only for baseline verification and migration comparison; remove the legacy environment, build dependencies and runtime support after modern acceptance passes.
- Retain baseline results, hashes and frozen comparison fixtures as verification evidence; do not retain a legacy fallback or maintain a second runtime.
- After that gate, attempt the newest stable Python and dependency releases available at execution time; prereleases require an explicit separate experiment.
- All subsequent phases use `.venv-modern` and its committed lockfile; Linux EC2 training uses a separately verified CUDA environment with the same application version and Python minor.
- Do not silently downgrade the modern stack. Record compatibility blockers, isolate optional native dependencies, and re-run parity checks after any port.
- Run migration parity on CPU first; validate MPS and CUDA separately before using their results.
- Use metres, seconds, radians, and Cartesian East–North `(x, y)` coordinates in the rebuilt simulation.
- Keep one episode clock and continuous vessel state across intermediate goals.
- Freeze scenarios independently of planners; pair initial conditions and exogenous randomness across comparisons.
- Keep perceived observations separate from scoring truth; report collisions, failures, safety interventions and missed deadlines honestly.
- Use AWS CLI for EC2 retraining operations; choose account, region, resource sizes and an explicit spending limit when executing that phase.
- Do not implement literature work, academic report writing, or competition presentation preparation in these plans.

---

All implementation paths below are relative to `rebuild/`. Run commands there. These are instructions and reference snippets, not installed software or benchmark results.

## Prerequisites, file map and decision gate

Finish baseline/migration and the classical/pretrained benchmarks first. The SARL checkpoint can be reused only with its original feature/action adapter. New marine observations and action semantics require a new policy; do not reshape weights and call that retraining. PPO is a reproducible modern baseline, not a claim that it is the newest or best maritime algorithm. The research review motivates prediction-aware inputs and constraints; this phase measures whether they help.

Create `src/shipnav/training/__init__.py`, `env.py`, `train.py`, `checkpoints.py`; `tests/test_training_env.py`, `test_checkpoint.py`; `tools/extract_episode_engine.py`; `cloud/prepare_launch.py`, `cloud/run-training.sh`, `cloud/local/`; `environments/ec2/pyproject.toml`, `uv.lock`. Modify `simulation.py`, `policies.py`, `service.py`, GUI model selection and model manifests. Cloud-local account/resource configuration and generated launch JSON stay ignored. Training code and environment locks are versioned.

### T1 — Extract a step API without changing episode behavior

The current loop already centralizes truth, observations, filtering and dynamics. Refactor it into a coroutine that pauses at the nominal-action boundary. The following transformation generates the first equivalent engine from the completed plan 03/03b code; review the generated diff, then maintain ordinary Python source. It avoids duplicating the motion/scoring logic in a second simulator.

File: `rebuild/tools/extract_episode_engine.py`

```python
from pathlib import Path

p=Path('src/shipnav/simulation.py')
s=p.read_text()
s=s.replace('def run_episode(', 'def episode_steps(',1)
old='nominal = tuple(policy(position,velocity,route[index],neighbours,radius,speed,step))'
new="""perception_ms = (perf_counter()-began)*1000
        nominal, controller_ms = yield {'position':position, 'velocity':velocity,
            'goal':route[index], 'neighbours':neighbours, 'radius':radius,
            'speed':speed, 'dt':step, 'observed':perceived, 'vessel':vessel,
            't':time, 'goal_index':index, 'step_cost': float(bool(diagnostics) and (diagnostics[-1].get('ship_collision',False) or diagnostics[-1].get('land_collision',False)))}
        nominal = tuple(nominal)
        began = perf_counter()"""
if old not in s:
    raise RuntimeError('Simulator action boundary changed; review extraction before applying')
s=s.replace(old,new,1)
s=s.replace('latencies.append((perf_counter()-inference_start)*1000)', 'latencies.append(float(controller_ms))')
s=s.replace("'decision_ms': (perf_counter()-began)*1000", "'decision_ms': perception_ms+controller_ms+(perf_counter()-began)*1000")
needle="'inference_ms': latencies, 'frames': frames, 'diagnostics': diagnostics}"
replacement="""'inference_ms': latencies, 'frames': frames, 'diagnostics': diagnostics,
            '_terminal_packet': {'position':position, 'velocity':velocity,
                'goal':route[min(index,len(route)-1)], 'neighbours':[],
                'radius':radius,'speed':speed,'dt':dt,'t':time,'goal_index':min(index,len(route)-1),
                'vessel':vessel,'observed':observer.observe(traffic,time,len(frames)-1),
                'step_cost':float(status=='collision')}}"""
if needle not in s: raise RuntimeError('Terminal schema changed')
s=s.replace(needle,replacement,1)
s += """

def run_episode(sea,route,traffic,policy,dt=.25,limit=100,radius=.5,speed=1.,cancel=lambda:False,**kwargs):
    engine=episode_steps(sea,route,traffic,policy,dt,limit,radius,speed,cancel,**kwargs)
    try:
        packet=next(engine)
        while True:
            started=perf_counter()
            action=policy(packet['position'],packet['velocity'],packet['goal'],packet['neighbours'],radius,speed,packet['dt'])
            cost=(perf_counter()-started)*1000
            packet=engine.send((action,cost))
    except StopIteration as ended:
        result=ended.value
        result.pop('_terminal_packet',None)
        return result
"""
p.write_text(s)
```

- [ ] Before extraction, save deterministic frames/status/diagnostics excluding timing for fixed direct/SARL/MPC scenarios in both dynamics tiers. Run the script, repeat them, and assert equality; run every simulation test. Ensure the `set_context` hook stays in the coroutine before yielding. The included transformation resumes with `(action, controller_ms)` and excludes learner pauses from decision latency. Verify that a deliberate learner sleep changes neither controller latency nor episode physics.
- [ ] Add `episode_steps` tests for a one-step collision, timeout, cancellation, intermediate-goal continuity and initially terminal episode. Closing a generator must release resources. Keep `run_episode` as the GUI/CLI compatibility driver.

### T2 — Gymnasium observation/action/reward contract

`reset(seed=...)` selects a frozen training scenario deterministically. The observation contains ego velocity, goal offset, heading/speed, nearest perceived targets with age/uncertainty and masks, and 16 map-clearance rays. It never includes future target truth. Observation normalization must be fixed from the training split or saved with a fitted normalizer. The initial implementation uses physically scaled raw features with finite bounds checked by the environment checker; record units and use separate normalization before larger runs if needed.

The action is a normalized two-dimensional desired velocity. It passes through the same marine tracker and optional safety filter used during evaluation. A zero vector requests braking; it does not teleport velocity to zero. Reward is final-goal progress minus time and terminal penalties; a per-step collision/domain cost is logged separately for the constraint ablation. Termination is success/collision; time limit is truncation. Evaluation freezes model, normalization and penalty multiplier.

File: `rebuild/src/shipnav/training/env.py`

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from math import dist,sin,cos,pi,hypot
from shipnav.maps import SeaMap
from shipnav.planning import astar,smooth
from shipnav.simulation import episode_steps
from shipnav.scenarios import load_traffic
from shipnav.observations import Observer

class NavigationEnv(gym.Env):
    def __init__(self,scenarios,filtered=True,limit=100.,penalty=0.):
        if not scenarios: raise ValueError('Empty training split')
        self.scenarios,self.filtered,self.limit,self.penalty=scenarios,filtered,limit,penalty
        self.action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(6+12*8+16,),dtype=np.float32)
        self.engine=None

    def encode(self,packet):
        p=packet['position']; v=packet['velocity']; g=packet['goal']; vessel=packet['vessel']
        values=[g[0]-p[0],g[1]-p[1],*v,sin(vessel.heading),cos(vessel.heading)]
        targets=sorted(packet['observed'],key=lambda o:dist(o['position'],p))[:12]
        for o in targets:
            values += [o['position'][0]-p[0],o['position'][1]-p[1],*o['velocity'],o['radius'],o['age'],o['margin'],1.]
        values += [0.]*(8*(12-len(targets)))
        for k in range(16):
            direction=(cos(k*pi/8),sin(k*pi/8)); lo,hi=0.,20.
            for _ in range(12):
                mid=(lo+hi)/2; q=tuple(x+mid*d for x,d in zip(p,direction))
                if self.sea.clear(p,q,.5): lo=mid
                else: hi=mid
            values.append(lo)
        return np.asarray(values,dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if self.engine is not None: self.engine.close()
        scene=self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        self.sea=SeaMap.from_dict(scene['map']); self.final=tuple(scene['goal'])
        route=smooth(self.sea,astar(self.sea,tuple(scene['start']),self.final))
        observer=Observer(seed=int(self.np_random.integers(2**31)),noise=.1,delay=.5,dropout=.1)
        traffic=load_traffic(scene)
        if scene.get('traffic_mode')=='reactive':
            from shipnav.reactive import ReactiveTraffic
            traffic=[ReactiveTraffic(ship,self.sea) for ship in traffic]
        elif scene.get('traffic_mode')!='scripted':
            raise ValueError('Unsupported training traffic mode')
        self.engine=episode_steps(self.sea,route,traffic,None,limit=self.limit,
                                  filtered=self.filtered,dynamics='marine',observer=observer)
        try: self.packet=next(self.engine)
        except StopIteration as error: raise ValueError('Training split contains initially terminal scenario') from error
        self.done=False; self.previous=dist(self.packet['position'],self.final)
        return self.encode(self.packet),{}

    def step(self,action):
        if self.done: raise RuntimeError('Reset after termination')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all(): raise ValueError('Invalid action')
        action=np.clip(action,-1,1); action=action/max(1.,hypot(*action))
        try:
            self.packet=self.engine.send((tuple(action),0.))
            status='running'; position=self.packet['position']; result=None
        except StopIteration as ended:
            result=ended.value; self.packet=result.pop('_terminal_packet')
            status=result['status']; position=self.packet['position']
        current=dist(position,self.final)
        reward=self.previous-current-.01
        cost=float(self.packet['step_cost'])
        reward += 10.*(status=='success')-10.*cost-self.penalty*cost
        self.previous=current
        terminated=status in ('success','collision')
        truncated=status in ('timeout','cancelled')
        self.done=terminated or truncated
        info={'cost':cost,'status':status}
        if result is not None:
            info['episode_result']=result
        return self.encode(self.packet),float(reward),terminated,truncated,info

    def close(self):
        if self.engine is not None: self.engine.close()
```

File: `rebuild/tests/test_training_env.py`

```python
from gymnasium.utils.env_checker import check_env
from shipnav.training.env import NavigationEnv

def scene():
    return {'map':{'bounds':[0,0,24,24],'land':[]},'start':[2,2],'goal':[20,2],
            'seed':1,'traffic':[],'traffic_mode':'scripted'}

def test_gym_seed_and_truncation():
    env=NavigationEnv([scene()],limit=.25)
    a,_=env.reset(seed=1); b,_=env.reset(seed=1)
    assert (a==b).all()
    _,_,terminated,truncated,info=env.step([0,0])
    assert not terminated and truncated and info['status']=='timeout'
    check_env(NavigationEnv([scene()]),skip_render_check=True)
```

- [ ] Run `python -m pytest tests/test_training_env.py -v` before/after implementation, then SB3 `check_env` and a 2,048-step local learning smoke test. Add one collision with `terminated=True,truncated=False`, terminal observation correctness, normalization save/load, and train/test split rejection tests.
- [ ] Assert the engine-generated terminal packet includes final ego/target observations, age, goal index and time. The coroutine strips this private packet from JSON episode results. The environment checker alone does not verify correct terminal semantics.
- [ ] Extend cost to the declared per-step domain exposure budget only after adding a truth-based current-step cost output from the engine; keep cost out of policy inputs. If training uses reactive scenarios, wrap targets exactly as in `service.execute`; reject unsupported scenario modes instead of silently treating them as scripted.

### T3 — Train, checkpoint and evaluate independent seeds

Create the first training entry point below. It is an executable local smoke/baseline runner. Before EC2 full runs, complete the checkpoint state extension and bounded-chunk loop described afterward.

File: `rebuild/src/shipnav/training/train.py`

```python
from pathlib import Path
import argparse,json
from stable_baselines3 import PPO
from shipnav.training.env import NavigationEnv

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenarios',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    scenarios=json.loads(args.scenarios.read_text())
    env=NavigationEnv(scenarios)
    model=PPO.load(args.resume,env=env,device=args.device) if args.resume else PPO(
        'MlpPolicy',env,seed=args.seed,device=args.device,n_steps=512,batch_size=64,verbose=1)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        model.learn(total_timesteps=args.steps,reset_num_timesteps=args.resume is None)
        model.save(args.output/'policy')
    finally:
        env.close()
```

File: `rebuild/src/shipnav/training/checkpoints.py`

```python
from pathlib import Path
import os,random,json
import numpy as np
import torch

def save_training_state(model,directory,metadata,penalty=0.):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    model.save(directory/'policy.tmp.zip')
    os.replace(directory/'policy.tmp.zip',directory/'policy.zip')
    state={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'penalty':penalty,'num_timesteps':model.num_timesteps}
    torch.save(state,directory/'state.tmp.pt')
    os.replace(directory/'state.tmp.pt',directory/'state.pt')
    # Write manifest last; upload only complete checkpoint directories.
    from hashlib import sha256
    metadata=dict(metadata,files={name:sha256((directory/name).read_bytes()).hexdigest() for name in ('policy.zip','state.pt')})
    (directory/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))

def update_penalty(current,mean_episode_cost,budget,rate=.05):
    return max(0.,current+rate*(mean_episode_cost-budget))
```

File: `rebuild/tests/test_checkpoint.py`

```python
from shipnav.training.checkpoints import update_penalty

def test_penalty_rises_on_budget_violation_and_is_nonnegative():
    assert update_penalty(1.,2.,1.)>1.
    assert update_penalty(0.,0.,1.)==0.
```

- [ ] Run `python -m pytest tests/test_checkpoint.py -v`; save/reload a real tiny PPO model and compare deterministic actions on fixed observations. SB3 saves policy/optimizer state; separately save Python/NumPy/torch/CUDA RNG, environment RNG/state or an explicit episode-boundary reset seed, normalization, penalty multiplier, scenario split/hash, observation/action schema, code revision and lock hashes. Save to a new uniquely named checkpoint directory each time; publish manifest last. Verify hashes before loading. Only load local trusted RNG pickle state with `weights_only=False`.
- [ ] Implement `--resume` state restoration with `random.setstate`, `np.random.set_state`, `torch.set_rng_state`, and CUDA state setters. Restore environment state/RNG, or resume at a recorded episode boundary and label it restart-equivalent rather than bitwise-identical. A model-only `PPO.load` is insufficient evidence of reproducible resume. SIGTERM requests a save at the next safe boundary; configure a wall-clock maximum independent of timesteps.
- [ ] Train fixed-budget chunks (e.g. 20,480 timesteps), save after each chunk, update the penalty using **completed episode costs** from that chunk, then continue with `reset_num_timesteps=False`. Keep ordinary PPO (`penalty=0`) and the Lagrangian-penalty experiment as separate variants. This pragmatic ablation is not a formal constrained-MDP solution or certified safety method; record budget violations even if reward improves.
- [ ] Pilot CPU versus one GPU on identical scenarios/steps and evaluate environment steps/s, learner utilization, memory and cost per million steps. PPO with a small MLP can be rollout-bound; select EC2 from measurements rather than assuming the largest GPU is best.
- [ ] Train at least three independent seeds; select checkpoints using development performance/cost only. Freeze normalization and penalty before held-out test. Include both safety-filter on/off evaluations and the same dynamics/perception used for classical comparators. Store complete training curves and seed-level outcomes.
- [ ] Add a `ModernLearned` adapter in `policies.py` that loads `policy.zip` plus manifest, validates action/observation/dynamics schema, builds exactly `NavigationEnv.encode` inputs from the current engine packet, and returns `model.predict(observation,deterministic=True)[0]` clipped/scaled by the same action transform. Extend service/GUI policy choices with `ppo`; select model directory containing this manifest. Test that a saved policy gives identical actions through Gym and the service adapter; reject incompatible manifests visibly instead of accepting arbitrary weights.

## T4 — AWS CLI EC2 execution, from discovery through teardown

Use AWS CLI v2 for every AWS operation in this phase, as requested. This document does not launch an instance. At execution, select the AWS profile/account, region, total spend limit, maximum wall time, checkpoint destination, and smallest resource meeting the pilot measurements. These values are not known from the workspace; collect them before a paid launch. Singapore is the user's timezone, not permission to assume an AWS region.

### Discover and prepare

- [ ] Verify CLI/session, then inspect regional candidates and quota. The commands below are read-only; `AWS_PROFILE` and `AWS_REGION` must identify the selected existing session.

```bash
aws --version
aws sts get-caller-identity
aws ec2 describe-instance-types --filters Name=current-generation,Values=true --output json > cloud/local/instance-types.json
aws ec2 describe-instance-type-offerings --location-type availability-zone --output json > cloud/local/offerings.json
aws service-quotas list-service-quotas --service-code ec2 --output json > cloud/local/ec2-quotas.json
```

Create `cloud/local/` before redirecting. Select an available x86_64 CPU instance or single NVIDIA GPU instance based on pilot memory/throughput. A current generation flag alone is not a recommendation; do not hardcode a supposedly newest G/P family or AMI. Start with On-Demand for the compatibility pilot; consider Spot only after interruption/resume testing.

- [ ] Query current regional On-Demand price with AWS Pricing CLI (`aws pricing get-products --service-code AmazonEC2 --region us-east-1` plus exact instanceType, regionCode, Linux, Shared, Used filters); save the JSON quote and timestamp. Calculate `hours × hourly rate + EBS + S3/requests + transfer + IPv4/NAT/endpoints` in a small Python budget script. Include stopped-instance EBS costs. A budget alert alone is not a spending cap; set a hard training timer and an independent shutdown schedule, allowing upload time.
- [ ] Resolve an official current base NVIDIA-driver DLAMI using the documented public SSM parameter in the selected region. First inspect [DLAMI release notes and ID discovery](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html); obtain the exact current parameter path there, store it as `SHIPNAV_AMI_PARAMETER`, and run:

```bash
aws ssm get-parameter --name "$SHIPNAV_AMI_PARAMETER" --query Parameter.Value --output text
aws ec2 describe-images --image-ids "$SHIPNAV_AMI_ID" --query 'Images[0].{Id:ImageId,Owner:OwnerId,Arch:Architecture,Root:RootDeviceName,Name:Name,Created:CreationDate}'
```

Set `SHIPNAV_AMI_ID` to the verified first command result. Check owner, architecture, root-device name and release/driver before launch. For a CPU pilot choose a current official Linux AMI instead. Install the modern locked application into its own environment; a DLAMI's preinstalled Python/PyTorch is not automatically the project's newest stack. Select a CUDA wheel/driver combination from the official PyTorch selector and validate it on the GPU.

- [ ] Reuse an appropriate subnet/security group/profile when available, after inspecting them. Require SSM access, no inbound SSH rule, IMDSv2 and encrypted EBS. The instance role needs `AmazonSSMManagedInstanceCore` plus least-privilege access only to this run's S3 prefix (and its KMS key if used). Provide HTTPS network access to SSM/S3/package sources via documented routes/endpoints; private networking can incur NAT/endpoint costs. Do not put access keys in user data or files uploaded to S3. Save IAM/network choices with the launch record.

### Build a concrete launch request

Create the script below. Required arguments are discovered resource identifiers, not invented IDs. It writes JSON only; it does not launch anything. Test request contents locally before paid execution.

File: `rebuild/cloud/prepare_launch.py`

```python
import argparse,json
from pathlib import Path

def request(ami,instance_type,subnet,group,profile,root,run_id,hours):
    if hours<=0: raise ValueError('Positive wall time required')
    tags=[{'Key':'Project','Value':'shipnav'},{'Key':'RunId','Value':run_id},{'Key':'MaxHours','Value':str(hours)}]
    return {'ImageId':ami,'InstanceType':instance_type,'MinCount':1,'MaxCount':1,
            'SubnetId':subnet,'SecurityGroupIds':[group],'IamInstanceProfile':{'Name':profile},
            'MetadataOptions':{'HttpTokens':'required','HttpEndpoint':'enabled','HttpPutResponseHopLimit':1},
            'BlockDeviceMappings':[{'DeviceName':root,'Ebs':{'VolumeSize':80,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],
            'InstanceInitiatedShutdownBehavior':'stop','ClientToken':run_id,
            'TagSpecifications':[{'ResourceType':r,'Tags':tags} for r in ('instance','volume')]}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('ami','instance-type','subnet','group','profile','root','run-id'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--hours',type=float,required=True)
    a=p.parse_args(); Path('cloud/local').mkdir(parents=True,exist_ok=True)
    Path('cloud/local/launch.json').write_text(json.dumps(request(a.ami,a.instance_type,a.subnet,a.group,a.profile,a.root,a.run_id,a.hours),indent=2))
```

- [ ] Test the pure `request()` with synthetic IDs: assert one instance, IMDSv2, no key pair, encrypted EBS, delete-on-termination and run tags; reject zero hours. The 80 GiB disk is a starting proposal: check selected AMI minimum size and measured disk use before emitting the actual request. MaxHours is only a tag; it does not enforce shutdown.
- [ ] Generate the concrete JSON from discovered IDs, inspect its resource/cost selection, then run the CLI permission check. `--dry-run` intentionally returns a nonzero `DryRunOperation` when permission is sufficient; distinguish that from `UnauthorizedOperation` and other failures.

```bash
aws ec2 run-instances --cli-input-json file://cloud/local/launch.json --dry-run
aws ec2 run-instances --cli-input-json file://cloud/local/launch.json --output json > cloud/local/launched.json
```

Do not execute the second command until execution-time region/resource/budget choices are settled. Reuse the same ClientToken when retrying the same request; do not launch duplicates on a timeout. Extract and record the returned instance ID, then `aws ec2 wait instance-status-ok --instance-ids "$SHIPNAV_INSTANCE_ID"` and inspect `aws ssm describe-instance-information` for that exact instance.

### Install, train and persist

- [ ] Upload a source archive containing only `rebuild/` implementation, lockfiles and frozen scenario assets to the private run prefix using `aws s3 cp`. Include SHA256 manifest. Do not upload ignored personal PDFs/slides/docs, local credentials or virtual environments. If using a Git archive, confirm the implementation commit exists first. The source/weights license determines what can be redistributed; keep the bucket private.
- [ ] Use `aws ssm send-command --document-name AWS-RunShellScript --instance-ids "$SHIPNAV_INSTANCE_ID" --parameters file://cloud/local/commands.json --output-s3-bucket-name "$SHIPNAV_BUCKET" --output-s3-key-prefix "$SHIPNAV_RUN_PREFIX/logs"`. Write the `commands` JSON array with Python `json.dumps`, never shell-concatenate untrusted values. Record returned CommandId and poll `get-command-invocation`; SSM dispatch success does not mean training succeeded.
- [ ] In the SSM script create a dedicated work directory, download/verify the source manifest, install the pinned uv runtime and matching Python, then run `uv sync --locked --extra train` against the verified Linux CUDA project. Run CPU/GPU smoke and checkpoint restore tests before the full job. Set `MPLBACKEND=Agg`; GUI/Qt/GDAL packages are not required on the training host unless a selected external backend needs them.
- [ ] Schedule an OS shutdown with `sudo shutdown -h +MINUTES` before training, with minutes computed from the approved wall time plus checkpoint grace. The instance shutdown behavior is stop. Also run training under `timeout --signal=TERM --kill-after=120s DURATION ...` and upload on each checkpoint, normal exit and termination. Configure an independent AWS-side stop schedule for a hung host when a firm cost bound is required; neither tags nor a budget alarm will stop it automatically.
- [ ] `cloud/run-training.sh` should run the bounded-chunk trainer, upload complete checkpoint directories with `aws s3 sync`, and write a final status/manifest after upload. Verify object hashes by downloading and checking manifest SHA256 values. Multipart S3 ETags are not file SHA256 hashes. Keep final/best checkpoints plus a bounded rolling history of resumable checkpoints; use a lifecycle rule for temporary artifacts only after defining retention.
- [ ] Record GPU/driver/CUDA, OS/AMI, instance type, region, lock/hash, code/split/seed, steps/s, runtime, estimated cost, SSM IDs and checkpoint hashes. Run one deliberate stop/resume pilot and local CPU model-load/action check before launching multiple seeds. No performance claim follows merely from a successful training process exit.

### Download, verify and clean up

```bash
aws s3 sync "s3://$SHIPNAV_BUCKET/$SHIPNAV_RUN_PREFIX/checkpoints" results/cloud-checkpoints
aws ec2 stop-instances --instance-ids "$SHIPNAV_INSTANCE_ID"
aws ec2 wait instance-stopped --instance-ids "$SHIPNAV_INSTANCE_ID"
```

- [ ] Verify downloaded hashes, load the model locally and run the held-out evaluation/GUI adapter tests. Inspect SSM training exit status and saved final manifest. If checkpoints are incomplete, retain the stopped disk until recovery is decided.
- [ ] After verified recovery, terminate only the recorded run instance with `aws ec2 terminate-instances --instance-ids "$SHIPNAV_INSTANCE_ID"` and wait for termination. Check attached volumes with run tags, snapshots, public IP allocations, schedules and any run-specific NAT/endpoints for residual charges. Delete only resources created for this run and no longer needed; retain chosen S3 model/evaluation artifacts. Cancel the independent stop schedule if one was created. Save cleanup evidence and actual runtime/cost estimate.

References: [EC2 launch CLI](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html), [regional offerings](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-type-offerings.html), [SSM command CLI](https://docs.aws.amazon.com/cli/latest/reference/ssm/send-command.html), [Gymnasium migration](https://gymnasium.farama.org/introduction/migration_guide/). Refresh AWS instance/AMI/pricing information at execution; no account or cloud resources were queried by this planning revision.

### Concrete bounded cloud runner

Use this after T3's full-state checkpoint/restore and termination handler are implemented. Set `--device cpu` for the CPU pilot. Chunks use unique local directories, avoiding a partially replaced published checkpoint. At startup/resume discover the last **verified complete** S3 checkpoint and use a new output root; never overwrite a previous run's chunk numbers. The timeout budget bounds training only; use the independent shutdown deadline to also bound downloads/uploads and a hung AWS CLI. Final upload failures must produce a failed run status.

File: `rebuild/cloud/run-training.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${SHIPNAV_PYTHON:?Path to locked EC2 Python required}"
: "${SHIPNAV_SCENARIOS:?Frozen training scenario JSON required}"
: "${SHIPNAV_OUTPUT:?Local checkpoint root required}"
: "${SHIPNAV_S3_URI:?Private run S3 prefix required}"
: "${SHIPNAV_SEED:?Training seed required}"
: "${SHIPNAV_CHUNKS:?Bounded chunk count required}"
: "${SHIPNAV_SECONDS:?Maximum runtime required}"
mkdir -p "$SHIPNAV_OUTPUT"
export MPLBACKEND=Agg
sync_artifacts() { aws s3 sync "$SHIPNAV_OUTPUT" "$SHIPNAV_S3_URI" --only-show-errors; }
trap sync_artifacts EXIT
start=$SECONDS
previous=''
for ((chunk=0; chunk<SHIPNAV_CHUNKS; chunk++)); do
    remaining=$((SHIPNAV_SECONDS-(SECONDS-start)))
    if ((remaining<=120)); then break; fi
    directory="$SHIPNAV_OUTPUT/chunk-$chunk"
    args=(--scenarios "$SHIPNAV_SCENARIOS" --output "$directory" --seed "$SHIPNAV_SEED" --steps 20480 --device cuda)
    if [[ -n "$previous" ]]; then args+=(--resume "$previous"); fi
    timeout --signal=TERM --kill-after=120s "$remaining" "$SHIPNAV_PYTHON" -m shipnav.training.train "${args[@]}"
    previous="$directory/policy.zip"
    sync_artifacts
 done
```

- [ ] Validate with `bash -n cloud/run-training.sh`; test one tiny CPU chunk and interrupted chunk on a local disposable directory before EC2. Capture stdout/stderr through SSM/S3. The script has no credentials and deliberately fails on missing run settings.
