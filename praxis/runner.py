"""CLI runner for PRAXIS experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .archive import PRAXISArchive, ProgramCandidate
from .data import build_test_loader, build_train_loader
from .generator import DiscreteGenerator
from .gpt import gpt_suggest_programs
from .models import build_resnet34
from .programs import (
    compile_program_from_spec,
    prog_argmax_first,
    prog_confidence_threshold,
    prog_entropy_gate,
    prog_even_bias,
    prog_low_index,
    prog_modulo3_time,
    prog_near_center,
    prog_odd_bias,
    prog_second_best,
    prog_top2_margin_switch,
)
from .training import (
    archive_metrics_on_probe,
    eval_accuracy,
    eval_over_universes,
    mixture_eval,
    objective_drift,
    symbolic_reward_term,
    train_solver_one_universe,
)
from .universe import Universe, assert_no_overlap
from .utils import DEVICE, set_seed


def _init_logs(args: argparse.Namespace, seed: int, universes: List[Universe]) -> Dict[str, Any]:
    return {
        "args": vars(args),
        "seed": seed,
        "universes": [u.to_dict() for u in universes],
        "generator": {
            "entropy": [],
            "max_prob": [],
            "u_star": [],
            "kl_to_prev": [],
            "temperature": [],
            "q": [],
            "symbolic_term": [],
        },
        "solver": {
            "train_ce": [],
            "train_reg": [],
            "alpha_t": [],
            "total_loss": [],
            "gate_total": [],
            "gate_avg_per_step": [],
            "gate_steps_nonzero": [],
            "gate_total_steps": [],
        },
        "archive": {
            "entropy": [],
            "max_w": [],
            "u_star": [],
            "margin": [],
            "F": [],
            "B": [],
            "U": [],
            "s": [],
            "w": [],
            "program_families": [],
        },
        "fixed_eval": {},
        "fixed_mean": [],
        "reward": [],
        "reward_components": {
            "perf": [],
            "symbolic": [],
            "lambda_ce_term": [],
            "lambda_reg_term": [],
        },
        "objective_drift": {
            "delta_obj": [],
            "delta_q_tv": [],
        },
        "gpt": {
            "events": [],
        },
        "robustness": {
            "min_acc": [],
            "cvar_acc": [],
            "mean_acc": [],
            "std_acc": [],
            "accs": [],
        },
        "heldout": {
            "min_acc": [],
            "cvar_acc": [],
            "mean_acc": [],
            "std_acc": [],
            "accs": [],
        },
        "timing": {
            "iter_total": [],
            "solver_train": [],
            "archive_update": [],
            "generator_update": [],
            "robust_eval": [],
            "heldout_eval": [],
        },
    }


def run(seed: int, args: argparse.Namespace) -> str:
    set_seed(seed)

    sizes = np.linspace(args.size_min, args.size_max, args.size_grid).astype(int).tolist()
    augs = np.linspace(args.aug_min, args.aug_max, args.aug_grid).tolist()

    corruptions = ["none", "blur", "jpeg", "noise"]
    severities = [0.0, 0.3, 0.6]

    universes = [
        Universe(s, a, c, sev)
        for s in sizes
        for a in augs
        for c in corruptions
        for sev in severities
    ]
    num_universes = len(universes)

    heldout_corruptions = ["blur", "jpeg", "noise"]

    heldout_universes = [
        Universe(s, a, c, sev)
        for s in args.heldout_sizes
        for a in args.heldout_augs
        for c in heldout_corruptions
        for sev in args.heldout_severities
    ]

    assert_no_overlap(universes, heldout_universes)

    generator = DiscreteGenerator(num_universes=num_universes).to(DEVICE)
    gen_opt = optim.Adam(generator.parameters(), lr=args.gen_lr)

    solver = build_resnet34(num_classes=args.num_classes).to(DEVICE)

    programs = [
        ProgramCandidate("ProgArgmaxFirst", prog_argmax_first, "base"),
        ProgramCandidate("ProgEvenBias", prog_even_bias, "base"),
        ProgramCandidate("ProgLowIndex", prog_low_index, "base"),
        ProgramCandidate("ProgSecondBest", prog_second_best, "rank"),
        ProgramCandidate("ProgTop2MarginSwitch", prog_top2_margin_switch, "rank"),
        ProgramCandidate("ProgConfidenceThreshold", prog_confidence_threshold, "threshold"),
        ProgramCandidate("ProgEntropyGate", prog_entropy_gate, "threshold"),
        ProgramCandidate("ProgOddBias", prog_odd_bias, "symmetry"),
        ProgramCandidate("ProgModulo3Time", prog_modulo3_time, "temporal"),
        ProgramCandidate("ProgNearCenter", prog_near_center, "distributional"),
    ]

    archive: Optional[PRAXISArchive]
    if args.disable_archive:
        archive = None
    else:
        archive = PRAXISArchive(
            programs,
            beta=args.beta,
            eta_p=args.eta_p,
            gamma=args.gamma,
            max_programs=args.max_programs,
        )

    fixed_tests = {
        s: build_test_loader(
            args.test_dir,
            size_test=args.size_test,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            seed=int(s),
        )
        for s in args.test_seeds
    }

    probe_loader = build_test_loader(
        args.test_dir,
        size_test=args.probe_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        seed=int(args.probe_seed),
    )

    logs = _init_logs(args, seed, universes)

    obj_per_u = np.zeros(num_universes, dtype=np.float64)
    q_prev = None

    if args.prior_type == "uniform":
        q_prior = torch.full((num_universes,), 1.0 / num_universes, device=DEVICE)
    else:
        q_prior = None

    temperature = args.temp_start

    for t in range(args.iters):
        iter_start = time.perf_counter()
        idxs, logp, q_t = generator.sample(args.group_size, temperature=temperature)
        q_np = q_t.detach().cpu().numpy().astype(np.float64)

        ent = float(-(q_np * np.log(q_np + 1e-12)).sum())
        maxp = float(q_np.max())
        u_star = int(q_np.argmax())

        if q_prev is None:
            kl_prev = 0.0
        else:
            kl_prev = float((q_np * (np.log(q_np + 1e-12) - np.log(q_prev + 1e-12))).sum())

        sym_term = symbolic_reward_term(archive) if archive is not None else 0.0

        logs["generator"]["entropy"].append(ent)
        logs["generator"]["max_prob"].append(maxp)
        logs["generator"]["u_star"].append(u_star)
        logs["generator"]["kl_to_prev"].append(kl_prev)
        logs["generator"]["temperature"].append(float(temperature))
        logs["generator"]["q"].append(q_np.tolist())
        logs["generator"]["symbolic_term"].append(float(sym_term))

        rewards = []
        ce_accum = []
        reg_accum = []
        tot_accum = []

        perf_terms = []
        sym_terms = []
        lam_ce_terms = []
        lam_reg_terms = []
        gate_total_list = []
        gate_avg_list = []
        gate_nonzero_list = []
        gate_steps_list = []

        solver_train_start = time.perf_counter()
        for ui in idxs:
            u = universes[ui]
            train_loader = build_train_loader(
                args.train_dir,
                size=u.size,
                aug_strength=u.aug_strength,
                corruption=u.corruption,
                severity=u.severity,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                seed=seed + 10 * t + ui,
            )

            if args.alpha_warmup_iters > 0:
                alpha_t = args.alpha * min(1.0, (t + 1) / args.alpha_warmup_iters)
            else:
                alpha_t = args.alpha

            solver_opt = optim.Adam(solver.parameters(), lr=args.solver_lr)

            train_stats = train_solver_one_universe(
                solver,
                train_loader,
                archive,
                alpha_t=alpha_t,
                steps=args.solver_steps,
                args=args,
                opt=solver_opt,
            )

            gate_total_list.append(train_stats.get("gate_total", 0))
            gate_avg_list.append(train_stats.get("gate_avg_per_step", 0.0))
            gate_nonzero_list.append(train_stats.get("gate_steps_nonzero", 0))
            gate_steps_list.append(train_stats.get("gate_total_steps", 0))

            ce_accum.append(train_stats["train_ce"])
            reg_accum.append(train_stats["train_reg"])
            tot_accum.append(train_stats["total_loss"])

            fixed_accs = [
                eval_accuracy(solver, test_loader, max_batches=args.max_test_batches)
                for test_loader in fixed_tests.values()
            ]
            fixed_mean = float(np.mean(fixed_accs))

            lam_ce_term = args.lambda_ce * float(train_stats["train_ce"])
            lam_reg_term = args.lambda_reg * float(train_stats["train_reg"])
            sym_add = args.sym_coef * float(sym_term)

            r = fixed_mean - lam_ce_term - lam_reg_term + sym_add
            rewards.append(r)

            perf_terms.append(float(fixed_mean))
            sym_terms.append(float(sym_add))
            lam_ce_terms.append(float(lam_ce_term))
            lam_reg_terms.append(float(lam_reg_term))

            obj_per_u[ui] = r
        logs["timing"]["solver_train"].append(time.perf_counter() - solver_train_start)

        logs["solver"]["train_ce"].append(float(np.mean(ce_accum)) if ce_accum else 0.0)
        logs["solver"]["train_reg"].append(float(np.mean(reg_accum)) if reg_accum else 0.0)
        logs["solver"]["total_loss"].append(float(np.mean(tot_accum)) if tot_accum else 0.0)
        logs["solver"]["alpha_t"].append(
            float(
                args.alpha
                if args.alpha_warmup_iters == 0
                else (args.alpha * min(1.0, (t + 1) / args.alpha_warmup_iters))
            )
        )
        logs["solver"]["gate_total"].append(int(np.mean(gate_total_list)) if gate_total_list else 0)
        logs["solver"]["gate_avg_per_step"].append(
            float(np.mean(gate_avg_list)) if gate_avg_list else 0.0
        )
        logs["solver"]["gate_steps_nonzero"].append(
            int(np.mean(gate_nonzero_list)) if gate_nonzero_list else 0
        )
        logs["solver"]["gate_total_steps"].append(
            int(np.mean(gate_steps_list)) if gate_steps_list else 0
        )

        logs["reward_components"]["perf"].append(float(np.mean(perf_terms)) if perf_terms else 0.0)
        logs["reward_components"]["symbolic"].append(
            float(np.mean(sym_terms)) if sym_terms else 0.0
        )
        logs["reward_components"]["lambda_ce_term"].append(
            float(np.mean(lam_ce_terms)) if lam_ce_terms else 0.0
        )
        logs["reward_components"]["lambda_reg_term"].append(
            float(np.mean(lam_reg_terms)) if lam_reg_terms else 0.0
        )

        iter_key = f"iter_{t + 1}"
        fixed_accs_by_seed = {
            str(s): float(eval_accuracy(solver, test_loader, max_batches=args.max_test_batches))
            for s, test_loader in fixed_tests.items()
        }
        logs["fixed_eval"][iter_key] = fixed_accs_by_seed
        logs["fixed_mean"].append(float(np.mean(list(fixed_accs_by_seed.values()))))

        mix_eval = mixture_eval(
            solver,
            universes,
            q_np,
            args.test_dir,
            args.batch_size,
            args.num_classes,
            seed,
            max_batches=3,
        )
        logs.setdefault("mixture_eval", []).append(mix_eval)

        if (t + 1) % args.robust_eval_every == 0:
            robust_start = time.perf_counter()
            accs = eval_over_universes(
                solver,
                universes,
                args.test_dir,
                args.batch_size,
                args.num_classes,
                seed,
                args.robust_size_test,
                args.robust_max_batches,
            )

            k = max(1, int(args.cvar_alpha * len(accs)))
            worst = np.sort(accs)[:k]

            logs["robustness"]["min_acc"].append(float(accs.min()))
            logs["robustness"]["mean_acc"].append(float(accs.mean()))
            logs["robustness"]["std_acc"].append(float(accs.std()))
            logs["robustness"]["cvar_acc"].append(float(worst.mean()))
            logs["robustness"].setdefault("accs", []).append(accs.tolist())

            logs["timing"]["robust_eval"].append(time.perf_counter() - robust_start)
        else:
            logs["timing"]["robust_eval"].append(0.0)
            logs["robustness"]["min_acc"].append(None)
            logs["robustness"]["mean_acc"].append(None)
            logs["robustness"]["std_acc"].append(None)
            logs["robustness"]["cvar_acc"].append(None)
            logs["robustness"]["accs"].append(None)

        if (t + 1) % args.heldout_eval_every == 0:
            heldout_start = time.perf_counter()
            accs = eval_over_universes(
                solver,
                heldout_universes,
                args.test_dir,
                args.batch_size,
                args.num_classes,
                seed + 999,
                args.heldout_size_test,
                args.heldout_max_batches,
            )

            k = max(1, int(args.cvar_alpha * len(accs)))
            worst = np.sort(accs)[:k]

            logs["heldout"]["min_acc"].append(float(accs.min()))
            logs["heldout"]["mean_acc"].append(float(accs.mean()))
            logs["heldout"]["std_acc"].append(float(accs.std()))
            logs["heldout"]["cvar_acc"].append(float(worst.mean()))
            logs["heldout"].setdefault("accs", []).append(accs.tolist())

            logs["timing"]["heldout_eval"].append(time.perf_counter() - heldout_start)
        else:
            logs["timing"]["heldout_eval"].append(0.0)
            logs["heldout"]["min_acc"].append(None)
            logs["heldout"]["mean_acc"].append(None)
            logs["heldout"]["std_acc"].append(None)
            logs["heldout"]["cvar_acc"].append(None)
            logs["heldout"]["accs"].append(None)

        archive_update_start = time.perf_counter()
        if archive is not None:
            F_vec, B_vec = archive_metrics_on_probe(
                solver, probe_loader, archive, max_batches=args.probe_batches
            )
            if args.freeze_archive:
                upd = {
                    "entropy": archive.entropy(archive.w),
                    "max_w": float(np.max(archive.w)),
                    "u_star": archive.programs[int(np.argmax(archive.w))].name,
                    "margin": 0.0,
                    "F": {},
                    "B": {},
                    "U": {},
                    "s": {},
                    "w": {archive.programs[i].name: float(archive.w[i]) for i in range(len(archive.w))},
                }
            else:
                upd = archive.update(F_vec, B_vec)
        else:
            upd = None
        logs["timing"]["archive_update"].append(time.perf_counter() - archive_update_start)

        if archive is not None and upd is not None:
            logs["archive"]["entropy"].append(float(upd["entropy"]))
            logs["archive"]["max_w"].append(float(upd["max_w"]))
            logs["archive"]["u_star"].append(upd["u_star"])
            logs["archive"]["margin"].append(float(upd["margin"]))
            logs["archive"]["F"].append(upd["F"])
            logs["archive"]["B"].append(upd["B"])
            logs["archive"]["U"].append(upd["U"])
            logs["archive"]["s"].append(upd["s"])
            logs["archive"]["w"].append(upd["w"])
            logs["archive"]["program_families"].append(
                {p.name: p.family for p in archive.programs}
            )
        else:
            logs["archive"]["entropy"].append(0.0)

        if archive is not None and upd is not None:
            w_vals = np.array(list(upd["w"].values()), dtype=np.float64)
            j_star = int(np.argmax(w_vals))
            w_star = float(w_vals[j_star])

            rel_logs = {
                name: float(np.log((wi + 1e-12) / (w_star + 1e-12)))
                for name, wi in upd["w"].items()
                if float(wi) > 0.0
            }
            logs["archive"].setdefault("log_w_ratio", []).append(rel_logs)
        else:
            logs["archive"].setdefault("log_w_ratio", []).append({})

        gpt_event = {"t": int(t), "enabled": bool(args.enable_gpt), "status": "skipped"}
        should_fire = (
            bool(args.enable_gpt)
            and archive is not None
            and upd is not None
            and ((t + 1) % int(args.gpt_every) == 0)
        )

        if should_fire:
            w_dict = upd["w"]
            u_dict = upd["U"]
            top = sorted(w_dict.items(), key=lambda kv: kv[1], reverse=True)[: min(5, len(w_dict))]
            topU = sorted(u_dict.items(), key=lambda kv: kv[1], reverse=True)[: min(5, len(u_dict))]
            ctx = {
                "iter": int(t + 1),
                "fixed_mean": float(logs["fixed_mean"][-1]),
                "solver_train_ce": float(logs["solver"]["train_ce"][-1]),
                "solver_train_reg": float(logs["solver"]["train_reg"][-1]),
                "gen_ent": float(ent),
                "gen_max": float(maxp),
                "archive_ent": float(upd["entropy"]),
                "archive_max_w": float(upd["max_w"]),
                "archive_u_star": str(upd["u_star"]),
                "top_programs_by_weight": [{"name": n, "w": float(w)} for n, w in top],
                "top_programs_by_utility": [{"name": n, "U": float(u)} for n, u in topU],
                "num_programs_current": int(len(archive.programs)),
                "request": {"N": int(args.gpt_max_new_programs)},
            }

            gpt_res = gpt_suggest_programs(
                enabled=True,
                provider=args.gpt_provider,
                model=args.gpt_model,
                temperature=args.gpt_temperature,
                max_tokens=args.gpt_max_tokens,
                max_new_programs=args.gpt_max_new_programs,
                context=ctx,
                num_classes=args.num_classes,
            )

            compiled_names = []
            compile_errors = []
            new_candidates = []
            if gpt_res.get("ok", False):
                for sp in gpt_res.get("parsed_program_specs", []):
                    try:
                        nm, fn, fam, meta = compile_program_from_spec(
                            sp, num_classes=args.num_classes
                        )
                        new_candidates.append(ProgramCandidate(nm, fn, fam, meta=meta))
                        compiled_names.append(nm)
                    except Exception as exc:
                        compile_errors.append(f"compile_failed: {type(exc).__name__}: {exc}")

            add_report = (
                archive.maybe_add_programs(new_candidates)
                if new_candidates
                else {"added": [], "skipped": [], "total": len(archive.programs)}
            )

            gpt_event = {
                "t": int(t),
                "enabled": True,
                "fired": True,
                "provider": str(args.gpt_provider),
                "model": str(args.gpt_model),
                "ok": bool(gpt_res.get("ok", False)),
                "status": str(gpt_res.get("status", "")),
                "errors": list(gpt_res.get("errors", [])) + compile_errors,
                "prompt": gpt_res.get("prompt", ""),
                "raw": gpt_res.get("raw", ""),
                "parsed_program_specs": gpt_res.get("parsed_program_specs", []),
                "compiled": compiled_names,
                "add_report": add_report,
            }

        logs["gpt"]["events"].append(gpt_event)

        generator_update_start = time.perf_counter()
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        adv = rewards_t - rewards_t.mean()
        pg_loss = -(logp * adv.detach()).mean()

        ent_t = -(q_t * torch.log(q_t + 1e-12)).sum()
        kl_prev_term = torch.tensor(0.0, device=DEVICE)
        kl_prior_term = torch.tensor(0.0, device=DEVICE)

        if args.kl_mode in ["prev", "both"] and q_prev is not None:
            q_prev_t = torch.tensor(q_prev, device=DEVICE)
            kl_prev_term = (q_t * (torch.log(q_t + 1e-12) - torch.log(q_prev_t + 1e-12))).sum()

        if args.kl_mode in ["prior", "both"]:
            if q_prior is None:
                q_prior = q_t.detach()
            kl_prior_term = (q_t * (torch.log(q_t + 1e-12) - torch.log(q_prior + 1e-12))).sum()

        loss = (
            pg_loss
            + args.kl_coef * kl_prev_term
            + args.kl_prior_coef * kl_prior_term
            - args.ent_coef * ent_t
        )

        if not args.freeze_generator:
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()
        logs["timing"]["generator_update"].append(time.perf_counter() - generator_update_start)

        logs["reward"].append(float(np.mean(rewards)) if rewards else 0.0)

        if q_prev is None:
            logs["objective_drift"]["delta_obj"].append(0.0)
            logs["objective_drift"]["delta_q_tv"].append(0.0)
        else:
            delta_obj, delta_tv = objective_drift(q_np, q_prev, obj_per_u)
            logs["objective_drift"]["delta_obj"].append(float(delta_obj))
            logs["objective_drift"]["delta_q_tv"].append(float(delta_tv))

        q_prev = q_np.copy()

        if args.temp_anneal > 0:
            temperature = max(args.temp_min, temperature * args.temp_anneal)

        logs["timing"]["iter_total"].append(time.perf_counter() - iter_start)

        print(
            " ".join(
                [
                    f"[Iter {t + 1}/{args.iters}]",
                    f"fixed_mean={logs['fixed_mean'][-1]:.3f}",
                    f"gen_ent={ent:.3f}",
                    f"gen_max={maxp:.3f}",
                    f"arch_ent={logs['archive']['entropy'][-1]:.3f}",
                    f"CE={logs['solver']['train_ce'][-1]:.3f}",
                    f"REG={logs['solver']['train_reg'][-1]:.3f}",
                    f"sym_raw={sym_term:.3f}",
                ]
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"logs_praxis_mit_indoor_seed{seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return str(out_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PRAXIS-aligned MIT Indoor (optional GPT archive injection)")
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="./logs")

    p.add_argument("--num-classes", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--size-min", type=int, default=400)
    p.add_argument("--size-max", type=int, default=2400)
    p.add_argument("--size-grid", type=int, default=5)
    p.add_argument("--aug-min", type=float, default=0.0)
    p.add_argument("--aug-max", type=float, default=1.0)
    p.add_argument("--aug-grid", type=int, default=5)

    p.add_argument("--size-test", type=int, default=800)
    p.add_argument("--max-test-batches", type=int, default=50)

    p.add_argument("--probe-size", type=int, default=400)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--probe-batches", type=int, default=5)

    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])

    p.add_argument("--solver-lr", type=float, default=1e-3)
    p.add_argument("--solver-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--alpha-warmup-iters", type=int, default=2)

    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--eta-p", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--max-programs", type=int, default=64)

    p.add_argument("--gen-lr", type=float, default=1e-2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--freeze-generator", action="store_true", help="Freeze generator q_t")

    p.add_argument("--temp-start", type=float, default=1.0)
    p.add_argument("--temp-anneal", type=float, default=0.95)
    p.add_argument("--temp-min", type=float, default=0.2)

    p.add_argument("--lambda-ce", type=float, default=0.0)
    p.add_argument("--lambda-reg", type=float, default=0.0)

    p.add_argument("--sym-coef", type=float, default=0.0)

    p.add_argument("--test-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    p.add_argument("--disable-archive", action="store_true")
    p.add_argument("--freeze-archive", action="store_true")

    p.add_argument("--enable-gpt", action="store_true")
    p.add_argument("--gpt-provider", type=str, default="openai", choices=["openai", "azure"])
    p.add_argument("--gpt-model", type=str, default="gpt-4o-mini")
    p.add_argument("--gpt-every", type=int, default=5)
    p.add_argument("--gpt-max-new-programs", type=int, default=4)
    p.add_argument("--gpt-temperature", type=float, default=0.4)
    p.add_argument("--gpt-max-tokens", type=int, default=800)

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--robust-eval-every", type=int, default=10)
    p.add_argument("--robust-size-test", type=int, default=800)
    p.add_argument("--robust-max-batches", type=int, default=3)
    p.add_argument("--cvar-alpha", type=float, default=0.10)

    p.add_argument("--heldout-sizes", type=int, nargs="+", default=[2000])
    p.add_argument("--heldout-augs", type=float, nargs="+", default=[0.5])
    p.add_argument("--heldout-severities", type=float, nargs="+", default=[0.15, 0.45, 0.75])
    p.add_argument("--heldout-eval-every", type=int, default=10)
    p.add_argument("--heldout-size-test", type=int, default=800)
    p.add_argument("--heldout-max-batches", type=int, default=3)

    p.add_argument("--kl-mode", type=str, choices=["prev", "prior", "both", "none"], default="prev")
    p.add_argument("--kl_prior_coef", type=float, default=0.02)
    p.add_argument("--prior-type", type=str, choices=["uniform", "initial_q"], default="uniform")

    p.add_argument(
        "--psi-gate",
        type=str,
        choices=["none", "low_conf", "high_disagree", "both"],
        default="none",
    )
    p.add_argument("--psi-conf-thresh", type=float, default=0.55)
    p.add_argument("--psi-disagree-thresh", type=float, default=0.4)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Device] {DEVICE}")

    all_logs = []
    for seed in args.seeds:
        path = run(seed, args)
        all_logs.append(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / "aggregate_praxis_mit_indoor.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump({"seed_logs": all_logs}, f, indent=2)

    print("Done. Wrote:\n- " + str(agg_path) + "\n- " + "\n- ".join(all_logs))
