# EXPO-FT on the Franka FR3 — runbook (100-episode online RL)

EXPO-FT (`pd-perry/expo-ft`, *Sample-Efficient RL Finetuning for VLAs*) is the
**weight-moving** baseline on the double-cable task, the counterpart to DSRL:

| | DSRL | EXPO-FT | SubRL |
|---|---|---|---|
| what RL controls | the flow-matching **latent** | the **action chunk** (edit + selection) | an action-space **residual** |
| pi0.5 weights | frozen | **LoRA fine-tuned online** | frozen |
| policy server | separate openpi serve | **none — the learner owns pi0.5** | separate serve |
| reward | operator keypress | operator keypress | VLM verifier |

Everything below is embodiment plumbing plus hyperparameters sized for **100 real
episodes**. `expo_ft/agents/alg/expo_ft.py` (the algorithm) is untouched except for
two things the port genuinely needs, both documented at their site:
`discount_power` (this buffer stores one transition per *decision*, not per env
step) and passing the model's own state into the output transform (our actions are
deltas; upstream's DROID actions are velocities).

Design counterpart for the DSRL port: `~/Desktop/dsrl_pi0/DSRL_FRANKA_RUNBOOK.md`.

---

## 0. One-time setup

### a. openpi must be OUR fork, branch `Franka_EXPO`

EXPO-FT needs `Pi0.sample_actions(num_samples=N)` — N candidate chunks off ONE
prefix forward. Our branch adds that (and the batched `Policy.infer`) **on top of**
the station's existing DSRL/SubRL wire layers, and carries the RL TrainConfig
`expo_pi05_franka_double_cable_r6_wcrop_lora`. Do not use `pd-perry/openpi`: it has
none of our data configs, and the DROID configs it does have are the wrong robot.

```bash
cd ~/Desktop/expo_ft
git clone -b Franka_EXPO https://github.com/Destiny000621/openpi.git expo_ft/agents/vla/openpi
uv sync                     # server (learner) venv: ./.venv
```

`client/` is upstream's DROID rollout client. **We do not use it** — the robot half
is avantbot. Never run `client/uv sync`; it exists only so upstream's scripts still
read.

Verify the sampler before anything else (CPU, seconds):

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  expo_ft/agents/vla/openpi/src/openpi/models/pi0_expo_sampling_test.py -q   # 8 gates
```

### b. The SFT checkpoint and its assets

RL starts from the wcrop checkpoint the frozen baseline was measured with, and
normalizes with **that checkpoint's own baked norm stats** — the RL TrainConfig
points its `AssetsConfig` straight into `<checkpoint>/assets`. Copy the checkpoint
directory to the learner box and, if the path differs, override it:

```bash
--config.pi05_weight_loader_path=/path/to/pi05_franka_double_cable_100_wcrop_10k/params
--config.pi05_assets_dir=/path/to/pi05_franka_double_cable_100_wcrop_10k/assets
--config.pi05_asset_id=local/double_cable_100_r6_v21
```

The checkpoint is a FULL fine-tune and the RL model is a LoRA model: openpi's
`CheckpointWeightLoader` fills the missing adapters from the fresh init
(`missing_regex=".*lora.*"`), and `freeze_filter` then trains only those. That is
the intended path, not a workaround.

### c. Warm-start data: this station's own rollouts

The buffer is seeded from recorded pi0.5 **rollouts** (`data_log_eval_wcrop`),
not from teleop demos. They are on-policy for the checkpoint RL starts from, they
already look like the online data (same plant, cameras and episode structure), and
they bring recorded **failures** — the negative signal a success-only seed cannot
give a critic, and the signal the online run would otherwise buy with robot time.
Successes are what the actor learns from: the BC pool is success-only by config, so
nothing distils toward a failed rollout.

As of 2026-09-06 that directory holds **38 episodes, 15 of them successes**
(`metadata.json: success` and a `SUCCESS` marker file agree on every one).

```bash
rsync -a ~/Desktop/Haply_Franka/data_log_eval_wcrop <learner>:~/expo_seed/
# the learner reads it via --rollout_seed_dir (run_franka.sh: ROLLOUT_DIR=...)
```

Decoding the videos takes minutes, so `--seed_cache <path.pkl>` stores the
PREPROCESSED rows (224 px) and makes every later start instant.

Three conversions in that loader are load-bearing, and all three are gated by
`tests/test_franka_offline.py`:

* **gripper in radians.** The recorder stores `gripper_pos` in 0-1 (1 = open); the
  checkpoint expects knuckle radians (its own norm stats: state q99 = 0.7263,
  action q99 = 0.7927), so the loader applies the deploy client's own mapping
  `rad = (1 - pos) * 0.7929`. Seeding the raw 0-1 column would look completely
  reasonable and be wrong in every state.
* **side frame pre-resized, wrist frame raw** — exactly what the live client puts
  on the wire, so seeded and online pixels went through the same resampling.
* **rot6d from `ee_pose` / `target_pose`**, using openpi's own converter routine.
  A sanity number worth knowing: the commanded pose LEADS the measured pose by
  ~18 mm in these recordings, which is the impedance lead the demos were collected
  with — if that number ever comes out near zero or in metres, state and action are
  being read from different frames.

Seeding from the SFT LeRobot demos is still available
(`--seed_source lerobot --dataset_repo_id <repo>`), but the repo id is REQUIRED
there: the demo conversion that matches this checkpoint
(`local/double_cable_100_r6_v21`, 100 episodes / 115,284 frames) is not
necessarily the one sitting on a given box — the station currently has
`double_cable_100_r6_v22`, which is a **99-episode** re-conversion (115,666
frames), i.e. a different episode selection. Do not seed from it by accident.

### d. Offline gates

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_franka_offline.py -q   # 16 gates
# robot side, from ~/Desktop/Haply_Franka:
pixi run pytest tests/test_expo_agent.py -q                                    # 9 gates
```

---

### e. Learner-box procedure (H200-5), in order

```bash
cd <EXPO_ROOT>/code/expo_ft
EXPO_ROOT=<EXPO_ROOT> source scripts/franka/learner_env.sh     # paths + all 8 GPUs
# once per rollout set (~3.5 min, CPU only, NOT inside the learner — see the script):
python scripts/franka/build_seed_cache.py --rollout_dir $ROLLOUT_DIR --out $EXP/seed_rows.pkl
bash scripts/franka/run_franka.sh                                # loads the cache, listens
```

Three things on that box that look like bugs and are not:

* `~/.cache/openpi` and `~/.cache/huggingface` are symlinks onto the ephemeral
  local SSD; when its target is missing, norm-stat loading dies with a
  `FileNotFoundError` deep inside openpi. `learner_env.sh` points
  `OPENPI_DATA_HOME` at a real directory instead.
* `~/.profile` sources a file under that same missing path, so **login shells**
  (`bash -l`, `ssh -t`) print an error; use plain `bash`.
* decoding rollout video inside a process that has imported the learner's stack
  (lerobot brings a second libav) deadlocks silently — that is why seeding is a
  separate script, and why `--allow_inline_seeding` is off.

## 1. Two processes (there is no third — no serve)

```bash
# 1) learner (learner box; owns pi0.5, the critic and the edit policy)
cd ~/Desktop/expo_ft && bash scripts/franka/run_franka.sh

# 2) if the learner is on another machine, forward its port FROM the robot box —
#    WITH keepalives (a silently dead tunnel still accepts connections and hangs):
ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L 9112:localhost:9112 <learner-host>
curl -s localhost:9112/healthz      # must answer before the robot session starts

# 3) robot loop (station)
cd ~/Desktop/Haply_Franka/vendor/avantbot && pixi shell -e droid-openpi
python -m avantbot.collect --config policy/franka_pi05_ee_fr3_expo
```

Sanity check before touching the arm:

```bash
curl -s localhost:9112/healthz     # -> {"mode":"train","base_policy":true,"replan_steps":25,...}
```

The learner compiles the whole decision path (N-sample pi0.5 + critic + edit) at
startup, on a synthetic observation, and logs how long it took. If you skip that
line and the robot connects first, the first decision blocks for minutes and reads
as a hang.

**Remote learner: set `agent.image_codec: jpeg`** in the session YAML. The wrist
frame must go **uncropped** (the crop is a learner-side transform), which is 2.7 MB
per decision at 720p; JPEG q95 is ~40× smaller and the training videos were lossy
AV1 anyway. On a local learner leave it `raw`.

---

## 2. The operator loop

One decision = 25 of pi0.5's 50 rows = **0.83 s** at 30 Hz. An episode is capped at
2700 ticks (90 s ≈ 108 decisions).

| key | meaning |
|---|---|
| **1** | SUCCESS — reward 1 on the last decision, recording saved **with** the marker, then home |
| **0** | FAILURE — reward 0, terminal, recording saved **without** the marker, then home |
| **h** | ABORT — dropped from the buffer *and* the recording discarded |
| **r** | resume → opens the next episode |

Each episode: *watch → `1` or `0` → re-stage → `r`*. The agent drives the recorder
at **both** edges, so **never press `space`/`s`/`d` yourself** — START_STOP is a
toggle and a manual press desynchronises recorder state from episode state (both
2026-09-05 DSRL incidents were exactly that).

**Label failures with `0`, never `h`.** The critic learns what *not* to select from
failures. Use `h` only for episodes that must never be learned from — an
unrecoverable state, or a takeover (EXPO-FT's actor batch is success-only BC over
chunks its own critic selected; a human-driven chunk was never a candidate).

The arm holds pose for the duration of each decision (round trip + sampling). The
agent logs any decision over 1 s — if that becomes common, the link or the learner
is the problem, not the policy.

---

## 3. Hyperparameters for a 100-episode budget

Upstream's values unless the "why" column says otherwise.

| knob | value | why |
|---|---|---|
| `N` / `n_edit_samples` | **8 / 8** | upstream; 16 candidates scored per decision |
| `residual_mode` | **chunk_offset** | ONE offset per chunk, broadcast over the 25 rows. Upstream's per-row residual is right for DROID's cartesian *velocities* and wrong for absolute 30 Hz *position* chunks — live 2026-09-06: edited chunks stepped 30-40 mm per row (base policy 0.4 mm/row), rot6d left the unit sphere by 0.2, gripper jumped 0.8 rad inside a chunk. The offset keeps the base sample's shape and moves where it goes |
| `residual_action_dims` | **xyz** | no rot6d edits (independently edited entries are not a rotation) and no gripper edits (an out-of-distribution lever). Upstream's `residual_action_xyzg` is the DROID special case |
| `edit_scale` | **0.1** | tanh-bounded, NORMALIZED units: ≈ up to 5 mm of xyz offset per axis per chunk. Upstream's 0.2 doubles that; raise once the critic ranks candidates sensibly |
| `replan_steps` | **25** | half of pi0.5's 50-row chunk = 0.83 s. Upstream replans at half its horizon too (8 of 16). Must equal the session's `open_loop_horizon` |
| `residual_action_xyzg` | **False** | upstream's pick task disables rotation edits; an RJ45 insertion is exactly where wrist alignment decides the episode |
| `discount` | **0.99 per DECISION** | horizon ≈ 100 decisions ≈ one episode. Upstream's 0.99 per 10 Hz env step would be 0.78 per decision here — blind to the only reward the task has |
| `discount_power` | **1** | one buffer transition IS one decision (see `franka_replay_buffer.py`) |
| `num_updates` | **4** per episode | README guidance `env_steps / num_updates ≈ 20-30`; 108 decisions / 4 = 27 |
| `utd_ratio` | **20** | upstream |
| `batch_size` (critic) | **64** | upstream |
| `actor_batch_size` | **16** | the actor step is a pi0.5 backward pass; upstream ran both at 64 on 4 GPUs |
| `use_full_augmentation` | **False** (crop only) | rotate/colour-jitter fights a checkpoint trained on a fixed wrist crop |
| `seed_source` | **rollouts** | `data_log_eval_wcrop`, seeded at decision stride into the online buffer (upstream's `offline_ratio=0` path) |
| `rollout_include_failures` | **1** | recorded failures are critic data (rewards 0, terminal); the actor's BC pool stays success-only |
| `num_data` | **0 = all** | 38 rollout episodes -> roughly 1.7k seeded decisions |
| `min_episodes_before_update` | **1** | upstream waits for 10 collected episodes; with the demos seeded there is something to learn from at once, and robot episodes are the scarce resource |
| `buffer_capacity` | **20,000** | ~1.7k seeded + 100 × ~108 ≈ 12.5k. The buffer is a ring: undersize it and it eats its own seed |
| `max_episode_steps` | **2700** (90 s) | the 100 demos average 38.4 s; same cap as DSRL |

**The budget is 100 online robot episodes** (`--num_episodes 100`, the same as the
DSRL run) — the seeded rollouts are a warm start, not part of it. Expected totals:
**~12.5k transitions**, **~8,000 critic steps** and **~400 pi0.5 LoRA steps**.

Memory: a stored transition is three 224² uint8 views (~450 KB — the zero
right-wrist slot pi0.5 pads with is stored too, upstream's layout), so a full run is
~7 GB of host RAM plus the preallocated ring.

---

## 4. Evaluation

```bash
EVAL_BASE_ONLY=0 bash scripts/franka/run_eval.sh          # the trained policy
EVAL_BASE_ONLY=1 bash scripts/franka/run_eval.sh          # frozen-pi0.5 baseline row
# robot side:
python -m avantbot.collect --config policy/franka_pi05_ee_fr3_expo_eval
```

Eval mode inserts nothing, updates nothing and saves nothing; the running success
tally the learner prints per episode **is** the eval. `--eval` requires `--resume 1`
and the trained run's `--run_name`, so an eval can never quietly score a fresh
random critic.

---

## 5. Resume, and what is on disk

`<output_dir>/<run_name>/`:

* `checkpoints/` — orbax: pi0.5 (LoRA) + critic + edit actor + temperature, saved
  every 5 episodes and on SIGINT. **Ctrl+C is safe.**
* `rows/ep_XXXXX_{0,1}.pkl` — each episode's *preprocessed* transitions (224 px,
  normalized). Restoring re-inserts the identical arrays, with no second pass
  through the transforms; the `_1` suffix carries the success label.
* `counters.json` — episode/update/success counters.

`--resume 1` restores all three. Restoring rows without a checkpoint gives you the
data but fresh weights — the learner says so rather than pretending otherwise.

---

## 6. Measured on 2026-09-06 (H200-5 learner, station robot, SSH tunnel)

| what | value |
|---|---|
| learner startup (cache present) | ~1.5 min: 12 GB checkpoint ×2 in 3.6 s each, 3,894 seed rows in 1.8 s, decision path compiled in ~10 s |
| VRAM at rest | 33 GB per GPU (pi0.5 + target copy + LoRA optimizer + critic) |
| decision, learner-side | **80-95 ms** (N=8 base + 8 edits + REDQ argmax + transforms) |
| decision, from the station | **209 ms JPEG** / 438 ms raw — RTT is 80 ms, JPEG payload 138 KB, raw 2.9 MB. Needs the HTTP/1.1 keep-alive both sides now have; without it a fresh connection per decision cost 485 ms |
| update block, 4 updates × UTD 20 | **38 s on 4 GPUs** (the budget on this shared box); 22 s on 8, 132 s on one. First block ever: ~270 s (JIT) |
| seed cache build | 38 rollouts in 3.4 min sequential; 1.8 GB; 3,894 decisions, 929 in the success pool |
| Ctrl+C / SIGTERM | saves a checkpoint (verified: step 8 written on kill) |
| chunk smoothness, same fixed observation ×14 | per-row residual (upstream): **4/14 smooth**, edited chunks 30-40 mm/row; chunk-offset xyz @ 0.1: **14/14 smooth**, 0.3-0.7 mm/row = the serve's own base chunks |
| 20-update offline warm start | 411 s on 4 GPUs (first update is the JIT; the rest ~7 s each) |

Optional, needs root on the STATION: `sysctl -w net.ipv4.tcp_slow_start_after_idle=0`
keeps the tunnel's congestion window warm between decisions (they are 833 ms
apart, longer than one RTO) and should trim a few tens of ms more.

## 7. Open items before the first live run

1. **H200-5 layout is final at `/mnt/localssd/Sichang`** (done 2026-09-06): `code/expo_ft`
   is a checkout of `Destiny000621/expo-ft:franka-port` with openpi
   `Destiny000621/openpi:Franka_EXPO` under `expo_ft/agents/vla/openpi`; the SFT
   checkpoint sits in `home_migrated/physical/.cache/openpi/hf/` (which is where
   `~/.cache/openpi` points, so the TrainConfig's HOME-relative default resolves);
   `seed/`, `logs/expo_franka/` (incl. `seed_rows.pkl`), and `run_remote.sh`. Launch:
   `bash /mnt/localssd/Sichang/run_remote.sh <log> <cmd...>`. **Use GPUs 0-3 only**
   (`learner_env.sh` default) — the box is shared. `/mnt/localssd` is not a separate
   disk at the moment (a directory on the 886 GB root fs), and `~/.profile` still
   sources a missing `/mnt/localssd/Sichang/env`, so use non-login shells.
2. **Start a fresh `--run_name`** for the real 100 episodes; `bringup*` runs hold a
   few synthetic probe episodes.
3. **`select_ratio_with_residual`** in wandb is the health metric for the edit
   policy: if the critic never picks an edited candidate, EXPO-FT has degenerated
   into best-of-N sampling from the SFT policy.
4. **Do not run the first episodes on a random critic.** With `updates == 0`
   the argmax over 16 candidates is a coin flip, and the live run picked an
   edited (noisier) candidate 10 times in 14. `--initial_updates N` pre-trains
   the critic on the seeded rollouts before the robot connects (≈ 38 s per update
   on 4 GPUs); 20-30 is a sensible warm start and is NOT upstream behaviour.
5. **The seed rollouts were recorded at the eval session's chunking** (replan every
   15 rows), so their executed chunks are slightly off the replan-25 cadence the
   online run uses. Inherent to seeding; the online data is exact.
