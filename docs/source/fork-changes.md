# What this fork changes

This is [profdocpizza/BindCraft2](https://github.com/profdocpizza/BindCraft2), a fork of
[PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2). It tracks upstream and adds:

1. a **net-charge budget** the ProteinMPNN redesign stage decodes under, so a campaign can require
   its sequences to carry a given charge instead of hoping a weight variant delivers one;
2. one **opt-in compilation knob** transferred from an Apache-2.0 optimisation kit, plus the
   measurements that say why nothing else from that kit was worth transferring;
3. a small **crash fix**: `use_cueq: true` now falls back instead of failing a campaign mid-flight
   on a host without Triton.

Base for everything below: upstream `a8d0f20` (2026-09-29). Measurements were taken on one
RTX 5090 (32 GB, sm_120, driver 580.173.02), Python 3.12, jax 0.11.2 with the cuda13 plugin.

---

## 1. A net-charge budget for the redesign stage

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `mpnn_target_charge` | `null` | Net charge every redesigned sequence must carry. `null` leaves the sampler exactly as upstream. |
| `mpnn_charge_tolerance` | `0` | Half-width of the accepted window. `0` asks for the exact integer. |
| `mpnn_charge_lambda_max` | `8.0` | Cap in logit units on the soft steering that precedes the hard mask. Raise it only to diagnose. |

```json
{
  "target": "hPDL1",
  "modality": "binder",
  "mpnn_target_charge": -8,
  "mpnn_charge_tolerance": 3
}
```

That asks for a net charge in `[-11, -5]` on every candidate the redesign stage emits. All three
settings take part in the design-identity hash, and all three are campaign-wide: a parameter sweep
cannot vary them per arm, because the MPNN model object is built once for the campaign.

### What it guarantees, and over what

**Net charge** counts Asp and Glu as −1, Lys and Arg as +1. **Histidine counts 0** — its charge at
the reported pH is partial and context dependent — and **free termini are not counted**. Both
choices keep the target an integer, which is what makes exactness meaningful. This is the same
quantity `rank.py` reports as `Binder_Net_Charge` (`K+R minus D+E`); it is *not* the pH-dependent
`Binder_Net_Charge` filter metric, which is a continuous Henderson-Hasselbalch reading of the same
sequence and will differ by a fraction of a charge unit.

**Scope** is every resolved residue of a chain that holds at least one designed position. So:

* the target chains never count — they hold no designed positions;
* a **fixed** residue inside the binder does count, which is what "this binder should have net
  charge −8" normally means, and matters when `redesign_interface` is false and the interface is
  held;
* a **tied** group contributes **once**, not once per state or copy, because the group is one
  physical residue decoded in several places. Multistate campaigns (one binder against human and
  mouse targets, say) therefore get the binder's charge, not twice it.

### How it works

No weights are retrained and no checkpoint is touched. At each autoregressive step the sampler
emits 21 amino-acid logits; two deterministic stages act on them, after the amino-acid bias and
before the Gumbel draw:

* **Lookahead reweighting.** The exact conditional is
  `p(s_t | s_<t, target) = p(s_t | s_<t) · P(target reachable | s_<=t) / Z`, so the logits are
  multiplied by the probability that the positions after this one can still supply whatever the
  target demands, estimated by a normal approximation to that remaining sum whose per-position mean
  and variance are read off the model's own marginals as decoding proceeds. This conditions the
  **endpoint** and nothing else.
* **Hard reachability pruning.** Any amino acid whose selection would put the target out of reach
  of the undecoded steps is masked. This, not the reweighting, is what makes the budget a guarantee.
  Pruning that would leave nothing to sample is dropped rather than emitting a degenerate
  distribution.

`log_probs`, and therefore the `score` each candidate reports, are computed from the **raw** logits,
so the number stays comparable between constrained and unconstrained runs.

An unreachable budget is refused before decoding, naming the achievable range:

```
ValueError: mpnn_target_charge 177 +-0 is unreachable: 127 designed position(s) over the
designed chains reach a net charge of -127 to 127
```

### What it costs

`tests/test_charge_constraint.py`, on the VHH scaffold this package ships (127 designed positions,
12 designs per row, temperature 0.1, `neutral` weights, CPU):

| condition | achieved charge | mean per-residue NLL |
|---|---|---|
| unconstrained | −5 to +1 (mean −1.8) | 0.726 |
| `mpnn_target_charge: -2` (its own preference) | −2 to −2 | 0.762 |
| `mpnn_target_charge: -8` | −8 to −8 | 0.798 |
| `mpnn_target_charge: 4` | +4 to +4 | 0.827 |
| `-10`, tolerance 2 | −10 to −8 | 0.795 |
| `-10`, tolerance 3 | −7 to −7 | 0.787 |
| `-8` with a fixed target chain in the complex | −8 to −8 | 1.213 |
| `-6` with half the binder held fixed | −6 to −6 | 0.976 |

Conditioning at the charge the model already prefers costs about +0.04 nats per residue, and a 6 e
shift about +0.07 to +0.10. The reference implementation's benchmark over 17 backbones reports the
same magnitude (+0.06 nats at the preferred charge, rising with the size of the shift).

Two things the table shows that are worth reading twice. The row with a fixed target chain has a
much higher NLL (1.213) because the score averages over every resolved residue including the
target's, not because the budget is expensive. And the tolerance rows land on the window edge
nearest the model's own preference rather than spreading across the window — the tolerance is a
deadband, so inside it the model is left alone and only the edge is enforced. Use the tolerance to
buy likelihood back, not to get a spread of charges.

### What is not claimed

* **No folding, expression or synthesis validation.** The cost above is the model's own likelihood,
  a cheap proxy for design quality rather than evidence of it. Whether a charge-shifted design still
  refolds to its backbone is what the campaign's own refold and filter stages are for.
* **Charge is not pI.** At a fixed net charge the isoelectric point still varies by more than a pH
  unit, because pI depends on the counts of ionisable residues and not only on their difference.
* **Reachability is an interval.** That is exact for charges in `{-1, 0, +1}` as long as every
  designed position keeps one uncharged amino acid in its alphabet. A bias or `omit` list that
  removes all fifteen uncharged amino acids somewhere would leave a lattice the interval does not
  describe; that case is refused at setup rather than missed silently.
* **Diversity under the budget is not characterised** beyond the composition shift implied above.

### Provenance

The controller is a port of the one in
[profdocpizza/LigandMPNN](https://github.com/profdocpizza/LigandMPNN), branch
`feature/constrained-decoding` (`constraints.py`, `docs/CONSTRAINED_DECODING.md`; LigandMPNN is MIT
licensed). That implementation is PyTorch and sits in the LigandMPNN codebase; **BindCraft2 does not
use LigandMPNN.** It ships its own JAX/Haiku ProteinMPNN in `bindcraft/mpnn/` with its own
checkpoints in `bindcraft/weights/`, so the two-stage controller was re-implemented against
`bindcraft/mpnn/sample.py` rather than vendored. Per-step schedules are built in
`bindcraft/proteinmpnn.py` and carried through `hk.scan` as scan inputs; the controller itself is
`bindcraft/mpnn/constraints.py`.

The reference implementation's other two constraints — an ε₂₈₀ floor and SPPS-motif steering — were
not ported. Both would fit the same machinery (ε₂₈₀ is another linear functional; SPPS risk is not
one and carries no guarantee), and neither is needed by the campaigns this fork runs.

### Tests

```bash
JAX_PLATFORMS=cpu python tests/test_charge_constraint.py        # the table above, ~1 min, CPU
bash tests/test_identical_to_upstream.sh upstream/main          # no budget set => byte-identical
```

The second test decodes a fixed scaffold at a fixed seed in a worktree of the base revision and in
the working tree, and diffs the sequences and scores byte for byte. It passes: with
`mpnn_target_charge` unset, the schedule is never built and the branch is resolved at trace time, so
the sampler is the upstream sampler.

---

## 2. Speed: what was transferred, and what turned out not to be worth it

The starting point was the ColabDesign optimisation kit in
[anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
at commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` (Apache-2.0, so commercially permissive and
compatible with redistribution here under attribution). Its `colabdesign/` kit makes BindCraft
**v1**'s design step faster — stock ColabDesign 1.1.3 with BindCraft at `efb5bfeb` — behind a
`--mode exact|fast`.

The first finding is that most of that kit is already in BindCraft2, implemented differently and in
one case better:

| Kit lever | What it does | Status in BindCraft2 `a8d0f20` |
|---|---|---|
| `triatt`, `pallas`, `proj`, `txla` | flash/Pallas attention kernels over AF2's shapes | **already present** — `bindcraft/af/accel.py` routes attention through `jax.nn.dot_product_attention` with a cuDNN backend, probed at start-up (`attention_backend: auto`), with a chunked fallback |
| `trimul` | fused triangle multiplication | **already present** — `accel.fused_triangle_multiplicative_update` via cuEquivariance (`use_cueq`) |
| `nosub` | drop sub-batch chunking above 384 tokens | **already present and better** — `subbatch_size: auto` drops chunking when the estimated design memory fits in half the free GPU memory (`design_workers.campaign_subbatch_size`), instead of dropping it unconditionally |
| `compilecache` | jax persistent compilation cache | **already present** — `bindcraft/__init__.py` sets `JAX_COMPILATION_CACHE_DIR` and the CLI honours an operator override |
| `hoist_prev`, `lowercache` | avoid per-trajectory re-tracing and re-lowering | **largely moot** — the kit's problem is that ColabDesign rebuilds `mk_afdesign_model` per trajectory; BindCraft2 builds its models once per campaign and caches compiled programs per shape (`CompiledModelCache`), so only the first trajectory of a process pays tracing |
| `parcompile` | XLA flags for parallel module compilation | **not present** — transferred, see below |
| `ln`, `opm_fold`, `transition` | Pallas LayerNorm / outer-product-mean / transition kernels | **not present** — not transferred, see below |

### Transferred: `BC2_PARALLEL_COMPILE` (opt-in, off by default)

`BC2_PARALLEL_COMPILE=1` appends `--xla_gpu_force_compilation_parallelism=min(16, usable CPUs)` to
`XLA_FLAGS` at import. This changes compilation scheduling only, so the executable and every number
it produces are unchanged.

It is **off by default because it measured slower here.** One minimal campaign
(`examples/pdl1.json --core benchmark`, one trajectory, one worker, one step per stage, empty
compile cache each time) on an otherwise idle host:

| `BC2_PARALLEL_COMPILE` | run 1 | run 2 | mean |
|---|---|---|---|
| `0` (default) | 72.8 s | 64.7 s | **68.8 s** |
| `1` | 99.4 s | 71.0 s | 85.2 s |

Two repeats per arm is a small sample, but both repeats point the same way and the effect is large
enough to keep the flag out of the default path. The plausible reading is that jax 0.11's XLA
already parallelises compilation adaptively and forcing 16 threads oversubscribes a machine that is
also running the design steps. On a host with more cores than this one, or a different XLA, it may
well win — hence the knob.

The kit also sets `--xla_gpu_enable_llvm_module_compilation_parallelism=true`. **The XLA in jax
0.11.2 does not know that flag**, and an unknown flag in `XLA_FLAGS` is a fatal abort at start-up
(`F...parse_flags_from_env.cc:234] Unknown flag in XLA_FLAGS`), not a warning. It is therefore not
set. That failure mode is the second reason this lever is opt-in: if a future jax drops
`--xla_gpu_force_compilation_parallelism` too, the default path is unaffected.

### Not transferred: the Pallas kernel levers

`ln`, `opm_fold` and `transition` (and the parts of `triatt` that BindCraft2's cuDNN path does not
already cover) bind kernels through the kit's shared `opt_core` package, and install themselves over
**ColabDesign's** haiku classes — several levers check the sha256 of the function body they replace
and stand aside if it is not the pinned one. BindCraft2 does not use ColabDesign; it carries its own
AlphaFold fork under `bindcraft/af/alphafold/`. Transferring those levers means re-targeting each
adapter at BindCraft2's module paths, taking `opt_core` as a dependency, and re-validating numerics
per kernel — days of work, not a patch transfer, and the headline win they report is against a stock
ColabDesign that has neither cuDNN attention nor a fused triangle multiplication, both of which
BindCraft2 already has. That trade did not look worth making; the door is open if the kernels are
ever wanted.

### `--core fused`: the optional cuEquivariance kernels, measured

`settings/core/fused.json` is a named profile that turns on `use_cueq`, the one accelerated path
BindCraft2 ships switched off:

```bash
bindcraft design my_campaign.json --core fused
```

On this host it is **slower**, which is why it is a profile and not a default. One trajectory,
`--core benchmark` seed, one worker, default stage step counts, warm compile cache:

| `use_cueq` | run 1 | run 2 |
|---|---|---|
| `false` (default) | 157.5 s | 146.2 s |
| `true` | 262.4 s | 167.1 s |

The first `use_cueq` run pays Triton's kernel compilation; the second still trails the default by
about 14%. The arms are not work-identical — the fused kernels move the numerics, so the two
trajectories diverge and can exit at different stages — so read this as "no gain here", not as a
precise penalty. The profile is kept because the cuEquivariance kernels are built for data-centre
parts and may well win on one.

### Fixed: `use_cueq: true` crashed a campaign mid-flight without Triton

`accel.cuequivariance_available()` tested only that `cuequivariance_jax` imports. Its triangle
kernels import **Triton** lazily, inside the kernel call, so on a host with `cuequivariance_jax` but
no Triton a campaign started, compiled, and died about 15 seconds into its first trajectory with
`ImportError: triton is required for GPU implementation`. The probe now requires Triton as well, so
`use_cueq: true` falls back to the stock path — the same way `attention_backend: auto` falls back
when the cuDNN probe fails — instead of failing the run. If you want the kernels, install Triton
(`pip install triton`) into the environment.

---

## Licensing

BindCraft2's source-available licence permits modification and derivative works, with prominent
notice of modifications; this page is that notice. The transferred material keeps its own terms:
the compilation lever comes from anthropics/uplifting-biomolecular-modeling (Apache-2.0) and the
charge controller from profdocpizza/LigandMPNN (MIT). Both are attributed above and in the source
files that carry them.
