"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import os
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_scheduled(self, tasks):
        dependents = [[] for _ in tasks]
        anti_dependents = [[] for _ in tasks]
        dep_count = [0] * len(tasks)
        for i, task in enumerate(tasks):
            dep_count[i] = len(task["deps"])
            for dep in task["deps"]:
                dependents[dep].append(i)
            for dep in task.get("anti_deps", ()):
                anti_dependents[dep].append(i)

        priority = [0] * len(tasks)
        downstream = [0] * len(tasks)
        use_anti_priority = int(os.environ.get("SCHED_ANTI_PRIORITY", "0"))
        propagate_priority_boost = int(
            os.environ.get("SCHED_PROPAGATE_PRIORITY_BOOST", "0")
        )
        for i in range(len(tasks) - 1, -1, -1):
            priority[i] = 1 + max(
                (priority[j] for j in dependents[i]), default=0
            )
            if propagate_priority_boost:
                priority[i] += tasks[i].get("priority_boost", 0)
            if use_anti_priority:
                priority[i] = max(
                    priority[i],
                    max((priority[j] for j in anti_dependents[i]), default=0),
                )
            downstream[i] = min(
                100000,
                len(dependents[i])
                + sum(downstream[j] for j in dependents[i])
                + (
                    len(anti_dependents[i])
                    + sum(downstream[j] for j in anti_dependents[i])
                    if use_anti_priority
                    else 0
                ),
            )

        ready = [i for i, count in enumerate(dep_count) if count == 0]
        ready_set = set(ready)
        scheduled = [False] * len(tasks)
        instrs = []
        remaining = len(tasks)

        engine_order = tuple(
            os.environ.get("ENGINE_ORDER", "load,valu,alu,store,flow").split(",")
        )
        assert set(engine_order) == {"load", "valu", "alu", "store", "flow"}
        tie_mode = os.environ.get("SCHED_TIE", "late")
        anti_backfill = int(os.environ.get("SCHED_ANTI_BACKFILL", "0"))
        anti_reserve = int(os.environ.get("SCHED_ANTI_RESERVE", "0"))
        use_alternatives = int(os.environ.get("SCHED_ALTERNATIVES", "0"))
        alt_min_valu_ready = int(os.environ.get("SCHED_ALT_MIN_VALU_READY", "7"))
        alt_start = int(os.environ.get("SCHED_ALT_START", "0"))
        alt_end = int(os.environ.get("SCHED_ALT_END", str(1 << 30)))
        alt_max_choices = int(os.environ.get("SCHED_ALT_MAX_CHOICES", str(1 << 30)))
        alt_event_whitelist = {
            x for x in os.environ.get("SCHED_ALT_EVENTS", "").split(";") if x
        }
        alt_compete_flow = int(os.environ.get("SCHED_ALT_COMPETE_FLOW", "0"))

        def ready_priority(i):
            return priority[i] + tasks[i].get("priority_boost", 0)

        def anti_ready(i):
            return all(scheduled[dep] for dep in tasks[i].get("anti_deps", ()))

        def slot_for_engine(i, engine, alt_to_flow):
            task = tasks[i]
            if i in alt_to_flow:
                if engine != "flow":
                    return None
                for alt_engine, alt_slot in task.get("alternatives", ()):
                    if alt_engine == engine:
                        return alt_slot
                return None
            if task["engine"] == engine:
                return task["slot"]
            return None

        alternative_choices = []
        while remaining:
            bundle = {}
            selected = []
            alt_to_flow = set()
            cycle = len(instrs)
            if (
                use_alternatives
                and len(alternative_choices) < alt_max_choices
                and alt_start <= cycle <= alt_end
            ):
                mandatory_flow_ready = any(
                    not scheduled[i]
                    and tasks[i]["engine"] == "flow"
                    and not tasks[i].get("alternatives")
                    and anti_ready(i)
                    for i in ready
                )
                valu_ready = [
                    i
                    for i in ready
                    if not scheduled[i]
                    and tasks[i]["engine"] == "valu"
                    and anti_ready(i)
                ]
                if (
                    (alt_compete_flow or not mandatory_flow_ready)
                    and len(valu_ready) >= alt_min_valu_ready
                ):
                    flexible = [
                        i
                        for i in valu_ready
                        if any(
                            engine == "flow"
                            for engine, _ in tasks[i].get("alternatives", ())
                        )
                        and (
                            not alt_event_whitelist
                            or tasks[i].get("alternative_tag") in alt_event_whitelist
                        )
                    ]
                    if flexible:
                        flexible.sort(
                            key=lambda i: (-ready_priority(i), -downstream[i], -i)
                        )
                        alt_to_flow.add(flexible[0])
            for engine in engine_order:
                limit = SLOT_LIMITS[engine]
                schedule_limit = limit
                if anti_reserve:
                    pending_same_cycle_writes = 0
                    for i in ready:
                        if (
                            scheduled[i]
                            or tasks[i]["engine"] != engine
                            or tasks[i].get("priority_boost", 0) <= 0
                        ):
                            continue
                        unresolved = [
                            dep
                            for dep in tasks[i].get("anti_deps", ())
                            if not scheduled[dep]
                        ]
                        if unresolved and all(
                            dep in ready_set and tasks[dep]["engine"] == "flow"
                            for dep in unresolved
                        ):
                            pending_same_cycle_writes += 1
                    schedule_limit = max(0, limit - pending_same_cycle_writes)
                candidates = [
                    i
                    for i in ready
                    if not scheduled[i]
                    and slot_for_engine(i, engine, alt_to_flow) is not None
                ]
                if tie_mode == "early":
                    candidates.sort(key=lambda i: (-ready_priority(i), i))
                elif tie_mode == "fanout":
                    candidates.sort(
                        key=lambda i: (-ready_priority(i), -downstream[i], -i)
                    )
                elif tie_mode == "low_pressure":
                    candidates.sort(
                        key=lambda i: (
                            -ready_priority(i),
                            len(tasks[i].get("anti_deps", ())),
                            -i,
                        )
                    )
                else:
                    candidates.sort(key=lambda i: (-ready_priority(i), -i))
                while len(bundle.get(engine, [])) < schedule_limit:
                    made_progress = False
                    for i in candidates:
                        if scheduled[i]:
                            continue
                        if any(
                            (not scheduled[dep] and dep not in selected)
                            for dep in tasks[i].get("anti_deps", ())
                        ):
                            continue
                        selected.append(i)
                        if i in alt_to_flow:
                            alternative_choices.append((cycle, i, engine))
                        bundle.setdefault(engine, []).append(
                            slot_for_engine(i, engine, alt_to_flow)
                        )
                        scheduled[i] = True
                        ready_set.remove(i)
                        made_progress = True
                        break
                    if not made_progress:
                        break

            if anti_backfill:
                for engine in engine_order:
                    limit = SLOT_LIMITS[engine]
                    if len(bundle.get(engine, [])) >= limit:
                        continue
                    candidates = [
                        i
                        for i in ready
                        if not scheduled[i]
                        and slot_for_engine(i, engine, alt_to_flow) is not None
                    ]
                    if tie_mode == "early":
                        candidates.sort(key=lambda i: (-ready_priority(i), i))
                    elif tie_mode == "fanout":
                        candidates.sort(
                            key=lambda i: (
                                -ready_priority(i),
                                -downstream[i],
                                -i,
                            )
                        )
                    elif tie_mode == "low_pressure":
                        candidates.sort(
                            key=lambda i: (
                                -ready_priority(i),
                                len(tasks[i].get("anti_deps", ())),
                                -i,
                            )
                        )
                    else:
                        candidates.sort(key=lambda i: (-ready_priority(i), -i))
                    while len(bundle.get(engine, [])) < limit:
                        made_progress = False
                        for i in candidates:
                            if scheduled[i]:
                                continue
                            if any(
                                (not scheduled[dep] and dep not in selected)
                                for dep in tasks[i].get("anti_deps", ())
                            ):
                                continue
                            selected.append(i)
                            if i in alt_to_flow:
                                alternative_choices.append((cycle, i, engine))
                            bundle.setdefault(engine, []).append(
                                slot_for_engine(i, engine, alt_to_flow)
                            )
                            scheduled[i] = True
                            ready_set.remove(i)
                            made_progress = True
                            break
                        if not made_progress:
                            break

            if not selected:
                raise RuntimeError("Scheduler made no progress")

            instrs.append(bundle)
            remaining -= len(selected)

            for i in selected:
                for dep in dependents[i]:
                    dep_count[dep] -= 1
                    if dep_count[dep] == 0 and dep not in ready_set:
                        ready.append(dep)
                        ready_set.add(dep)

            if len(ready) > 4096:
                ready = [i for i in ready if not scheduled[i]]
                ready_set = set(ready)

        self.schedule_alternative_choices = alternative_choices
        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Fully unrolled SIMD kernel for the submission shape.

        Values stay in scratch across all rounds. Tree positions are represented
        as absolute memory addresses so a lane can gather with load_offset
        without recomputing forest_values_p + idx every round.
        """
        assert batch_size % VLEN == 0

        tasks = []
        last_writer = {}
        last_readers = defaultdict(set)

        def add_task(
            engine,
            slot,
            reads=(),
            writes=(),
            alternatives=None,
            alternative_tag=None,
            priority_boost=0,
            extra_deps=(),
        ):
            reads = tuple(reads)
            writes = tuple(writes)
            deps = set()
            deps.update(extra_deps)
            for addr in reads:
                if addr in last_writer:
                    deps.add(last_writer[addr])
            for addr in writes:
                if addr in last_writer:
                    deps.add(last_writer[addr])
            anti_deps = set()
            for addr in writes:
                anti_deps.update(last_readers[addr])

            task_id = len(tasks)
            task = {
                "engine": engine,
                "slot": slot,
                "deps": deps,
                "anti_deps": anti_deps,
                "priority_boost": priority_boost,
            }
            if alternatives:
                task["alternatives"] = tuple(alternatives)
            if alternative_tag is not None:
                task["alternative_tag"] = alternative_tag
            tasks.append(task)

            write_set = set(writes)
            for addr in reads:
                if addr not in write_set:
                    last_readers[addr].add(task_id)
            for addr in writes:
                last_readers[addr].clear()
                last_writer[addr] = task_id
            return task_id

        def vec_op(op, dest, a, b, priority_boost=0):
            add_task(
                "valu",
                (op, dest, a, b),
                reads=tuple(range(a, a + VLEN)) + tuple(range(b, b + VLEN)),
                writes=range(dest, dest + VLEN),
                priority_boost=priority_boost,
            )

        def vec_madd(dest, a, b, c, priority_boost=0):
            add_task(
                "valu",
                ("multiply_add", dest, a, b, c),
                reads=(
                    tuple(range(a, a + VLEN))
                    + tuple(range(b, b + VLEN))
                    + tuple(range(c, c + VLEN))
                ),
                writes=range(dest, dest + VLEN),
                priority_boost=priority_boost,
            )

        def vec_select(dest, cond, a, b, priority_boost=0):
            add_task(
                "flow",
                ("vselect", dest, cond, a, b),
                reads=(
                    tuple(range(cond, cond + VLEN))
                    + tuple(range(a, a + VLEN))
                    + tuple(range(b, b + VLEN))
                ),
                writes=range(dest, dest + VLEN),
                priority_boost=priority_boost,
            )

        def alloc_vec(name):
            return self.alloc_scratch(name, VLEN)

        def load_scalar_const(addr, val):
            add_task("load", ("const", addr, val), writes=(addr,))

        def init_vec_const(name, val, keep_scalar=False):
            scalar = self.alloc_scratch(name + "_scalar")
            vec = alloc_vec(name)
            load_scalar_const(scalar, val)
            add_task(
                "valu",
                ("vbroadcast", vec, scalar),
                reads=(scalar,),
                writes=range(vec, vec + VLEN),
            )
            return (vec, scalar) if keep_scalar else vec

        def alu_lanes(op, dest, a, b):
            for lane in range(VLEN):
                add_task(
                    "alu",
                    (op, dest + lane, a + lane, b + lane),
                    reads=(a + lane, b + lane),
                    writes=(dest + lane,),
                )

        def alu_lanes_scalar(op, dest, a, b_scalar, priority_boost=0):
            for lane in range(VLEN):
                add_task(
                    "alu",
                    (op, dest + lane, a + lane, b_scalar),
                    reads=(a + lane, b_scalar),
                    writes=(dest + lane,),
                    priority_boost=priority_boost,
                )

        def scalar_parity(dest, val):
            for lane in range(VLEN):
                add_task(
                    "alu",
                    ("&", dest + lane, val + lane, one_s),
                    reads=(val + lane, one_s),
                    writes=(dest + lane,),
                )

        scratch_addr = self.alloc_scratch("scratch_addr")
        tree_frontier = self.alloc_scratch("tree_frontier", 4 * VLEN)
        d4_flow_pairs = {
            int(x)
            for x in os.environ.get("D4_FLOW_PAIRS", "").split(",")
            if x
        }
        d4_adaptive_flow_pairs = {
            int(x)
            for x in os.environ.get("D4_ADAPTIVE_FLOW_PAIRS", "").split(",")
            if x
        }
        d4_adaptive_flow_blocks = {
            int(x)
            for x in os.environ.get("D4_ADAPTIVE_FLOW_BLOCKS", "").split(",")
            if x
        }
        d4_adaptive_flow_phases = {
            x
            for x in os.environ.get("D4_ADAPTIVE_FLOW_PHASES", "final").split(",")
            if x
        }
        d4_adaptive_flow_events = set()
        for spec in os.environ.get("D4_ADAPTIVE_FLOW_EVENTS", "").split(";"):
            if spec:
                pair_s, block_s, phase = spec.split(":", 2)
                d4_adaptive_flow_events.add((int(pair_s), int(block_s), phase))
        assert d4_flow_pairs.isdisjoint(d4_adaptive_flow_pairs)
        assert len(d4_adaptive_flow_pairs) <= 4
        assert d4_flow_pairs | d4_adaptive_flow_pairs <= set(range(8))
        assert d4_adaptive_flow_phases <= {"early", "final"}
        d4_adaptive_copy_engine = os.environ.get(
            "D4_ADAPTIVE_COPY_ENGINE", "valu"
        )
        assert d4_adaptive_copy_engine in {"alu", "valu"}
        d4_optional_alternatives = int(
            os.environ.get("D4_OPTIONAL_ALTERNATIVES", "0")
        )
        d4_adaptive_storage = os.environ.get(
            "D4_ADAPTIVE_STORAGE", "frontier"
        )
        assert d4_adaptive_storage in {
            "frontier",
            "dedicated",
            "dedicated_diff",
        }
        d4_split_low_temps = int(os.environ.get("D4_SPLIT_LOW_TEMPS", "0"))
        assert 0 <= d4_split_low_temps <= 2
        d4_split_low_blocks = {
            int(x)
            for x in os.environ.get("D4_SPLIT_LOW_BLOCKS", "").split(",")
            if x
        }
        compact_tail_store_addrs = int(
            os.environ.get("COMPACT_TAIL_STORE_ADDRS", "0")
        )
        reuse_tree_store_addrs = int(
            os.environ.get("REUSE_TREE_STORE_ADDRS", "0")
        )
        reuse_tree_store_addrs_start = int(
            os.environ.get("REUSE_TREE_STORE_ADDRS_START", "29")
        )
        assert 0 <= reuse_tree_store_addrs_start <= 32
        if d4_adaptive_storage in {"dedicated", "dedicated_diff"}:
            assert compact_tail_store_addrs or reuse_tree_store_addrs
        alu_tree_broadcast_count = int(
            os.environ.get("ALU_TREE_BROADCAST_COUNT", "0")
        )
        tree_broadcast_zero_s = (
            self.alloc_scratch("tree_broadcast_zero_s")
            if alu_tree_broadcast_count
            or (
                d4_adaptive_flow_pairs
                and d4_adaptive_copy_engine == "alu"
                and d4_adaptive_storage != "dedicated_diff"
            )
            else None
        )
        load_scalar_const(scratch_addr, 7)
        for chunk in range(4):
            dest = tree_frontier + chunk * VLEN
            add_task(
                "load",
                ("vload", dest, scratch_addr),
                reads=(scratch_addr,),
                writes=range(dest, dest + VLEN),
            )
            if chunk != 3:
                add_task(
                    "flow",
                    ("add_imm", scratch_addr, scratch_addr, VLEN),
                    reads=(scratch_addr,),
                    writes=(scratch_addr,),
                )

        def tree_node_vec(name, abs_addr, keep_scalar=False):
            vec = alloc_vec(name)
            scalar = tree_frontier + abs_addr - 7
            if abs_addr >= 38 - alu_tree_broadcast_count:
                for lane in range(VLEN):
                    add_task(
                        "alu",
                        ("+", vec + lane, scalar, tree_broadcast_zero_s),
                        reads=(scalar, tree_broadcast_zero_s),
                        writes=(vec + lane,),
                    )
            else:
                add_task(
                    "valu",
                    ("vbroadcast", vec, scalar),
                    reads=(scalar,),
                    writes=range(vec, vec + VLEN),
                )
            return (vec, scalar) if keep_scalar else vec

        def tree_node_vec_sequence(prefix, start_abs, count):
            return [
                tree_node_vec(f"{prefix}_{i}", start_abs + i) for i in range(count)
            ]

        one_v, one_s = init_vec_const("one_v", 1, keep_scalar=True)
        two_v, two_s = init_vec_const("two_v", 2, keep_scalar=True)
        m4097_v = init_vec_const("m4097_v", 4097)
        m33_v = init_vec_const("m33_v", 33)
        m16896_v = init_vec_const("m16896_v", 16896)
        m9_v = init_vec_const("m9_v", 9)
        c0_v = init_vec_const("c0_v", 0x7ED55D16)
        c23_v = init_vec_const("c23_v", 0xE9F8CC1D)
        c2sh9_v = init_vec_const("c2sh9_v", 0xACCF6200)
        c4_v = init_vec_const("c4_v", 0xFD7046C5)
        sh16_v = init_vec_const("sh16_v", 16)
        sh19_v = init_vec_const("sh19_v", 19)
        depth4_base_v = init_vec_const("depth4_base_v", 22)
        add_even_v = init_vec_const("add_even_v", -6)
        add_odd_v = alloc_vec("add_odd_v")
        vec_op("+", add_odd_v, add_even_v, one_v)
        depth3_mask_v = alloc_vec("depth3_mask_v")
        vec_op("-", depth3_mask_v, one_v, add_even_v)

        c1_s = self.alloc_scratch("c1_s")
        c5_s = self.alloc_scratch("c5_s")
        load_scalar_const(c1_s, 0xC761C23C)
        load_scalar_const(c5_s, 0xB55A4F09)
        c5_v = alloc_vec("c5_v")
        add_task(
            "valu",
            ("vbroadcast", c5_v, c5_s),
            reads=(c5_s,),
            writes=range(c5_v, c5_v + VLEN),
        )

        root_node_v, root_node_s = tree_node_vec(
            "root_node_v", 7, keep_scalar=True
        )
        c5_root_s = self.alloc_scratch("c5_root_s")
        add_task(
            "alu",
            ("^", c5_root_s, c5_s, root_node_s),
            reads=(c5_s, root_node_s),
            writes=(c5_root_s,),
        )

        d1_nodes = [tree_node_vec(f"d1_node_{i}", 8 + i) for i in range(2)]
        for node in d1_nodes:
            vec_op("^", node, node, c5_v)
        d1_n0, d1_n1 = reversed(d1_nodes)

        d2_nodes = [tree_node_vec(f"d2_node_{i}", 10 + i) for i in range(4)]
        for node in d2_nodes:
            vec_op("^", node, node, c5_v)
        d2_n0, d2_n1, d2_diff0, d2_diff1 = reversed(d2_nodes)
        vec_op("-", d2_diff0, d2_diff0, d2_n0)
        vec_op("-", d2_diff1, d2_diff1, d2_n1)

        d3_nodes = [tree_node_vec(f"d3_node_{i}", 14 + i) for i in range(8)]
        for node in d3_nodes:
            vec_op("^", node, node, c5_v)
        (
            d3_n0,
            d3_n1,
            d3_diff_lo0,
            d3_diff_lo1,
            d3_n4,
            d3_n5,
            d3_diff_hi0,
            d3_diff_hi1,
        ) = reversed(d3_nodes)
        vec_op("-", d3_diff_lo0, d3_diff_lo0, d3_n0)
        vec_op("-", d3_diff_lo1, d3_diff_lo1, d3_n1)
        vec_op("-", d3_diff_hi0, d3_diff_hi0, d3_n4)
        vec_op("-", d3_diff_hi1, d3_diff_hi1, d3_n5)

        d4_nodes = tree_node_vec_sequence("d4_node", 22, 16)
        d4_even = [d4_nodes[2 * i] for i in range(8)]
        d4_odd_or_diff = [d4_nodes[2 * i + 1] for i in range(8)]
        d4_adaptive_raw = {}
        d4_prebuilt_diff = set()
        for alias_i, pair_i in enumerate(sorted(d4_adaptive_flow_pairs)):
            if d4_adaptive_storage == "dedicated_diff":
                raw = d4_odd_or_diff[pair_i]
                diff = alloc_vec(f"d4_adaptive_diff_{pair_i}")
                vec_op("-", diff, raw, d4_even[pair_i])
                d4_odd_or_diff[pair_i] = diff
                d4_adaptive_raw[pair_i] = raw
                d4_prebuilt_diff.add(pair_i)
                continue
            raw = (
                alloc_vec(f"d4_adaptive_raw_{pair_i}")
                if d4_adaptive_storage == "dedicated"
                else tree_frontier + alias_i * VLEN
            )
            if d4_adaptive_copy_engine == "alu":
                alu_lanes_scalar(
                    "+", raw, d4_odd_or_diff[pair_i], tree_broadcast_zero_s
                )
            else:
                vec_op("&", raw, d4_odd_or_diff[pair_i], d4_odd_or_diff[pair_i])
            d4_adaptive_raw[pair_i] = raw
        for i in range(8):
            if i not in d4_flow_pairs and i not in d4_prebuilt_diff:
                vec_op("-", d4_odd_or_diff[i], d4_odd_or_diff[i], d4_even[i])

        bit0 = alloc_vec("bit0")
        bit1 = alloc_vec("bit1")
        mix = alloc_vec("mix")
        pair = alloc_vec("pair")
        high_tmp = alloc_vec("high_tmp")
        d4_low_mix = alloc_vec("d4_low_mix") if d4_split_low_temps >= 1 else mix
        d4_low_pair = alloc_vec("d4_low_pair") if d4_split_low_temps >= 2 else pair

        n_vecs = batch_size // VLEN
        inp_values_p = 7 + n_nodes + batch_size

        vals = []
        idxs = []
        tmp0s = []
        tmp1s = []
        store_addrs = []
        for block in range(n_vecs):
            val_v = alloc_vec(f"val_{block}")
            idx_v = alloc_vec(f"idx_{block}")
            tmp0_v = alloc_vec(f"tmp0_{block}")
            tmp1_v = alloc_vec(f"tmp1_{block}")
            if reuse_tree_store_addrs and block >= reuse_tree_store_addrs_start:
                base = tree_frontier + block - reuse_tree_store_addrs_start
            elif compact_tail_store_addrs and block > 28:
                base = store_addrs[-1]
            else:
                base = self.alloc_scratch(
                    "tail_value_base"
                    if compact_tail_store_addrs and block == 28
                    else f"value_base_{block}"
                )
            if block == 0:
                load_scalar_const(base, inp_values_p)
            else:
                add_task(
                    "flow",
                    ("add_imm", base, store_addrs[-1], VLEN),
                    reads=(store_addrs[-1],),
                    writes=(base,),
                )
            add_task(
                "load",
                ("vload", val_v, base),
                reads=(base,),
                writes=range(val_v, val_v + VLEN),
            )
            vals.append(val_v)
            idxs.append(idx_v)
            tmp0s.append(tmp0_v)
            tmp1s.append(tmp1_v)
            store_addrs.append(base)

        tile_ids = list(range(n_vecs))
        tile_ids[29], tile_ids[30] = tile_ids[30], tile_ids[29]
        n_groups = 13
        stagger = 2
        group_ids = [
            tile_ids[g * n_vecs // n_groups : (g + 1) * n_vecs // n_groups]
            for g in range(n_groups)
        ]

        valu_c5_xor_blocks = {
            int(x)
            for x in os.environ.get("VALU_C5_XOR_BLOCKS", "30").split(",")
            if x
        }
        final_gather_priority_blocks = {
            int(x)
            for x in os.environ.get("FINAL_GATHER_PRIORITY_BLOCKS", "").split(",")
            if x
        }
        final_gather_priority_boost = int(
            os.environ.get("FINAL_GATHER_PRIORITY_BOOST", "0")
        )
        final_hash_tail_priority_blocks = {
            int(x)
            for x in os.environ.get(
                "FINAL_HASH_TAIL_PRIORITY_BLOCKS", ""
            ).split(",")
            if x
        }
        final_hash_tail_priority_boost = int(
            os.environ.get("FINAL_HASH_TAIL_PRIORITY_BOOST", "0")
        )
        final_hash_priority_blocks = {
            int(x)
            for x in os.environ.get("FINAL_HASH_PRIORITY_BLOCKS", "").split(",")
            if x
        }
        final_hash_priority_boost = int(
            os.environ.get("FINAL_HASH_PRIORITY_BOOST", "0")
        )
        hash3_alu_blocks = {
            int(x)
            for x in os.environ.get("HASH3_ALU_BLOCKS", "").split(",")
            if x
        }
        hash3_alu_rounds = {
            int(x)
            for x in os.environ.get("HASH3_ALU_ROUNDS", "").split(",")
            if x
        }
        final_hash_priority_map = {}
        for spec in os.environ.get("FINAL_HASH_PRIORITY_MAP", "").split(","):
            if spec:
                block_s, boost_s = spec.split(":", 1)
                final_hash_priority_map[int(block_s)] = int(boost_s)

        def get_final_hash_priority(block, round_i):
            if round_i != rounds - 1:
                return 0
            if block in final_hash_priority_map:
                return final_hash_priority_map[block]
            if block in final_hash_priority_blocks:
                return final_hash_priority_boost
            return 0

        def emit_hash(ids, final_xor_scalar=c5_s, round_i=None):
            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                vec_madd(
                    vals[block],
                    vals[block],
                    m4097_v,
                    c0_v,
                    priority_boost=hash_priority_boost,
                )

            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                vec_op(
                    ">>",
                    tmp0s[block],
                    vals[block],
                    sh19_v,
                    priority_boost=hash_priority_boost,
                )
                alu_lanes_scalar(
                    "^",
                    vals[block],
                    vals[block],
                    c1_s,
                    priority_boost=hash_priority_boost,
                )
            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                vec_op(
                    "^",
                    vals[block],
                    vals[block],
                    tmp0s[block],
                    priority_boost=hash_priority_boost,
                )

            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                vec_madd(
                    tmp0s[block],
                    vals[block],
                    m33_v,
                    c23_v,
                    priority_boost=hash_priority_boost,
                )
                vec_madd(
                    vals[block],
                    vals[block],
                    m16896_v,
                    c2sh9_v,
                    priority_boost=hash_priority_boost,
                )
            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                if block in hash3_alu_blocks and (
                    not hash3_alu_rounds or round_i in hash3_alu_rounds
                ):
                    alu_lanes(
                        "^", vals[block], vals[block], tmp0s[block]
                    )
                else:
                    vec_op(
                        "^",
                        vals[block],
                        vals[block],
                        tmp0s[block],
                        priority_boost=hash_priority_boost,
                    )

            for block in ids:
                hash_priority_boost = get_final_hash_priority(block, round_i)
                vec_madd(
                    vals[block],
                    vals[block],
                    m9_v,
                    c4_v,
                    priority_boost=hash_priority_boost,
                )

            for block in ids:
                tail_priority_boost = (
                    final_hash_tail_priority_boost
                    if round_i == rounds - 1
                    and block in final_hash_tail_priority_blocks
                    else 0
                )
                tail_priority_boost = max(
                    tail_priority_boost,
                    get_final_hash_priority(block, round_i),
                )
                vec_op(
                    ">>",
                    tmp0s[block],
                    vals[block],
                    sh16_v,
                    priority_boost=tail_priority_boost,
                )
                if final_xor_scalar is not None:
                    if (
                        final_xor_scalar == c5_s
                        and round_i == rounds - 1
                        and block in valu_c5_xor_blocks
                    ):
                        vec_op(
                            "^",
                            vals[block],
                            vals[block],
                            c5_v,
                            priority_boost=tail_priority_boost,
                        )
                    else:
                        alu_lanes_scalar(
                            "^",
                            vals[block],
                            vals[block],
                            final_xor_scalar,
                            priority_boost=tail_priority_boost,
                        )
            for block in ids:
                tail_priority_boost = (
                    final_hash_tail_priority_boost
                    if round_i == rounds - 1
                    and block in final_hash_tail_priority_blocks
                    else 0
                )
                tail_priority_boost = max(
                    tail_priority_boost,
                    get_final_hash_priority(block, round_i),
                )
                vec_op(
                    "^",
                    vals[block],
                    vals[block],
                    tmp0s[block],
                    priority_boost=tail_priority_boost,
                )

        def emit_parity(dest, val, use_scalar=False):
            if use_scalar:
                scalar_parity(dest, val)
            else:
                vec_op("&", dest, val, one_v)

        def round_root(
            ids,
            round_i,
            use_scalar_and,
            root_already_xored=False,
            use_valu_node_xor=False,
        ):
            if not root_already_xored:
                for block in ids:
                    vec_op("^", vals[block], vals[block], root_node_v)
            emit_hash(ids, final_xor_scalar=None, round_i=round_i)
            for block in ids:
                emit_parity(idxs[block], vals[block], use_scalar_and)
                vec_select(tmp1s[block], idxs[block], d1_n1, d1_n0)
                if use_valu_node_xor:
                    vec_op("^", vals[block], vals[block], tmp1s[block])
                else:
                    alu_lanes("^", vals[block], vals[block], tmp1s[block])

        def round_depth1(ids, round_i, use_scalar_and, use_valu_node_xor=False):
            emit_hash(ids, final_xor_scalar=None, round_i=round_i)
            for block in ids:
                emit_parity(tmp0s[block], vals[block], use_scalar_and)
                vec_select(tmp1s[block], tmp0s[block], d2_n1, d2_n0)
                vec_select(mix, tmp0s[block], d2_diff1, d2_diff0)
                vec_madd(tmp1s[block], idxs[block], mix, tmp1s[block])
                if use_valu_node_xor:
                    vec_op("^", vals[block], vals[block], tmp1s[block])
                else:
                    alu_lanes("^", vals[block], vals[block], tmp1s[block])
                vec_madd(idxs[block], idxs[block], two_v, tmp0s[block])

        def round_depth2(ids, round_i, use_scalar_and, use_valu_node_xor=False):
            emit_hash(ids, final_xor_scalar=None, round_i=round_i)
            for block in ids:
                emit_parity(tmp0s[block], vals[block], use_scalar_and)
                vec_op("&", bit0, idxs[block], one_v)
                vec_op(">>", bit1, idxs[block], one_v)
                vec_select(tmp1s[block], tmp0s[block], d3_n1, d3_n0)
                vec_select(mix, tmp0s[block], d3_diff_lo1, d3_diff_lo0)
                vec_madd(tmp1s[block], bit0, mix, tmp1s[block])
                vec_select(pair, tmp0s[block], d3_n5, d3_n4)
                vec_select(mix, tmp0s[block], d3_diff_hi1, d3_diff_hi0)
                vec_madd(pair, bit0, mix, pair)
                vec_select(tmp1s[block], bit1, pair, tmp1s[block])
                if use_valu_node_xor:
                    vec_op("^", vals[block], vals[block], tmp1s[block])
                else:
                    alu_lanes("^", vals[block], vals[block], tmp1s[block])
                vec_madd(idxs[block], idxs[block], two_v, tmp0s[block])

        final_depth4_select_blocks = {2, 4, 6, 8, 9, 10, 12, 17, 18, 19, 27, 29}
        early_depth4_select_blocks = {7, 18, 20, 21, 29, 30, 31}
        scalar_bit_depth4_blocks = {
            int(x)
            for x in os.environ.get("D4_SCALAR_BIT_BLOCKS", "").split(",")
            if x
        }
        scalar_bit0_depth4_blocks = scalar_bit_depth4_blocks | {
            int(x)
            for x in os.environ.get("D4_SCALAR_BIT0_BLOCKS", "").split(",")
            if x
        }
        scalar_bit1_depth4_blocks = scalar_bit_depth4_blocks | {
            int(x)
            for x in os.environ.get("D4_SCALAR_BIT1_BLOCKS", "").split(",")
            if x
        }
        scalar_bit2_depth4_blocks = scalar_bit_depth4_blocks | {
            int(x)
            for x in os.environ.get("D4_SCALAR_BIT2_BLOCKS", "").split(",")
            if x
        }
        d4_parity_dest_blocks = {
            int(x)
            for x in os.environ.get("D4_PARITY_DEST_BLOCKS", "").split(",")
            if x
        }
        d4_spill_val_blocks = {
            int(x)
            for x in os.environ.get("D4_SPILL_VAL_BLOCKS", "").split(",")
            if x
        }
        d4_spill_full_mix = int(os.environ.get("D4_SPILL_FULL_MIX", "0"))
        d4_spill_private_pair = int(
            os.environ.get("D4_SPILL_PRIVATE_PAIR", "0")
        )
        assert d4_spill_val_blocks <= final_depth4_select_blocks
        d4_priority_blocks = {
            int(x)
            for x in os.environ.get("D4_PRIORITY_BLOCKS", "").split(",")
            if x
        }
        d4_priority_pairs = {
            int(x)
            for x in os.environ.get("D4_PRIORITY_PAIRS", "0,1").split(",")
            if x
        }
        d4_priority_boost = int(os.environ.get("D4_PRIORITY_BOOST", "0"))

        def d4_pair_value(dest, block, pair_i, phase):
            pair_priority_boost = (
                d4_priority_boost
                if block in d4_priority_blocks and pair_i in d4_priority_pairs
                else 0
            )
            if pair_i in d4_flow_pairs:
                vec_select(
                    dest,
                    tmp0s[block],
                    d4_odd_or_diff[pair_i],
                    d4_even[pair_i],
                )
            elif (
                pair_i in d4_adaptive_flow_pairs
                and (
                    (
                        block in d4_adaptive_flow_blocks
                        and phase in d4_adaptive_flow_phases
                    )
                    or (pair_i, block, phase) in d4_adaptive_flow_events
                )
            ):
                if d4_optional_alternatives:
                    madd_slot = (
                        "multiply_add",
                        dest,
                        tmp0s[block],
                        d4_odd_or_diff[pair_i],
                        d4_even[pair_i],
                    )
                    select_slot = (
                        "vselect",
                        dest,
                        tmp0s[block],
                        d4_adaptive_raw[pair_i],
                        d4_even[pair_i],
                    )
                    add_task(
                        "valu",
                        madd_slot,
                        reads=(
                            tuple(range(tmp0s[block], tmp0s[block] + VLEN))
                            + tuple(
                                range(
                                    d4_odd_or_diff[pair_i],
                                    d4_odd_or_diff[pair_i] + VLEN,
                                )
                            )
                            + tuple(
                                range(
                                    d4_adaptive_raw[pair_i],
                                    d4_adaptive_raw[pair_i] + VLEN,
                                )
                            )
                            + tuple(
                                range(d4_even[pair_i], d4_even[pair_i] + VLEN)
                            )
                        ),
                        writes=range(dest, dest + VLEN),
                        alternatives=(
                            ("valu", madd_slot),
                            ("flow", select_slot),
                        ),
                        alternative_tag=f"{pair_i}:{block}:{phase}",
                        priority_boost=pair_priority_boost,
                    )
                else:
                    vec_select(
                        dest,
                        tmp0s[block],
                        d4_adaptive_raw[pair_i],
                        d4_even[pair_i],
                    )
            else:
                vec_madd(
                    dest,
                    tmp0s[block],
                    d4_odd_or_diff[pair_i],
                    d4_even[pair_i],
                    priority_boost=pair_priority_boost,
                )

        def precompute_depth4_value(block, phase):
            lookup_priority_boost = (
                d4_priority_boost if block in d4_priority_blocks else 0
            )
            spill_val = phase == "final" and block in d4_spill_val_blocks
            spill_store_task = None
            if spill_val:
                spill_store_task = add_task(
                    "store",
                    ("vstore", store_addrs[block], vals[block]),
                    reads=(store_addrs[block],)
                    + tuple(range(vals[block], vals[block] + VLEN)),
                )
            use_split_low = d4_split_low_temps and (
                not d4_split_low_blocks or block in d4_split_low_blocks
            )
            work_mix = (
                vals[block] if spill_val and d4_spill_full_mix else mix
            )
            low_mix = vals[block] if spill_val else (
                d4_low_mix if use_split_low else mix
            )
            low_pair = tmp0s[block] if spill_val else (
                d4_low_pair if use_split_low else pair
            )
            if block in scalar_bit0_depth4_blocks:
                alu_lanes_scalar(
                    "&",
                    bit0,
                    idxs[block],
                    one_s,
                    priority_boost=lookup_priority_boost,
                )
            else:
                vec_op(
                    "&",
                    bit0,
                    idxs[block],
                    one_v,
                    priority_boost=lookup_priority_boost,
                )
            if block in scalar_bit1_depth4_blocks:
                alu_lanes_scalar(
                    ">>",
                    bit1,
                    idxs[block],
                    one_s,
                    priority_boost=lookup_priority_boost,
                )
                alu_lanes_scalar(
                    "&",
                    bit1,
                    bit1,
                    one_s,
                    priority_boost=lookup_priority_boost,
                )
            else:
                vec_op(
                    ">>",
                    bit1,
                    idxs[block],
                    one_v,
                    priority_boost=lookup_priority_boost,
                )
                vec_op(
                    "&",
                    bit1,
                    bit1,
                    one_v,
                    priority_boost=lookup_priority_boost,
                )

            if spill_val and d4_spill_private_pair:
                if block in scalar_bit2_depth4_blocks:
                    alu_lanes_scalar(
                        ">>",
                        tmp1s[block],
                        idxs[block],
                        two_s,
                        priority_boost=lookup_priority_boost,
                    )
                else:
                    vec_op(
                        ">>",
                        tmp1s[block],
                        idxs[block],
                        two_v,
                        priority_boost=lookup_priority_boost,
                    )

                private_mix = vals[block]
                private_pair = idxs[block]
                d4_pair_value(private_mix, block, 7, phase)
                d4_pair_value(private_pair, block, 6, phase)
                vec_select(
                    high_tmp,
                    bit0,
                    private_pair,
                    private_mix,
                    priority_boost=lookup_priority_boost,
                )

                d4_pair_value(private_mix, block, 5, phase)
                d4_pair_value(private_pair, block, 4, phase)
                vec_select(
                    private_mix,
                    bit0,
                    private_pair,
                    private_mix,
                    priority_boost=lookup_priority_boost,
                )
                vec_select(
                    high_tmp,
                    bit1,
                    private_mix,
                    high_tmp,
                    priority_boost=lookup_priority_boost,
                )

                d4_pair_value(private_mix, block, 3, phase)
                d4_pair_value(private_pair, block, 2, phase)
                vec_select(
                    private_pair,
                    bit0,
                    private_pair,
                    private_mix,
                    priority_boost=lookup_priority_boost,
                )

                d4_pair_value(private_mix, block, 1, phase)
                d4_pair_value(tmp0s[block], block, 0, phase)
                vec_select(
                    private_mix,
                    bit0,
                    tmp0s[block],
                    private_mix,
                    priority_boost=lookup_priority_boost,
                )
                vec_select(
                    private_mix,
                    bit1,
                    private_mix,
                    private_pair,
                    priority_boost=lookup_priority_boost,
                )
                vec_select(
                    tmp1s[block],
                    tmp1s[block],
                    private_mix,
                    high_tmp,
                    priority_boost=lookup_priority_boost,
                )
                add_task(
                    "load",
                    ("vload", vals[block], store_addrs[block]),
                    reads=(store_addrs[block],),
                    writes=range(vals[block], vals[block] + VLEN),
                    extra_deps=(spill_store_task,),
                )
                return

            d4_pair_value(work_mix, block, 7, phase)
            d4_pair_value(pair, block, 6, phase)
            vec_select(
                tmp1s[block],
                bit0,
                pair,
                work_mix,
                priority_boost=lookup_priority_boost,
            )

            d4_pair_value(work_mix, block, 5, phase)
            d4_pair_value(pair, block, 4, phase)
            vec_select(
                work_mix,
                bit0,
                pair,
                work_mix,
                priority_boost=lookup_priority_boost,
            )
            vec_select(
                tmp1s[block],
                bit1,
                work_mix,
                tmp1s[block],
                priority_boost=lookup_priority_boost,
            )

            d4_pair_value(work_mix, block, 3, phase)
            d4_pair_value(pair, block, 2, phase)
            vec_select(
                high_tmp,
                bit0,
                pair,
                work_mix,
                priority_boost=lookup_priority_boost,
            )

            d4_pair_value(low_mix, block, 1, phase)
            if spill_val or (
                phase == "final" and block in d4_parity_dest_blocks
            ):
                d4_pair_value(tmp0s[block], block, 0, phase)
                vec_select(
                    low_mix,
                    bit0,
                    tmp0s[block],
                    low_mix,
                    priority_boost=lookup_priority_boost,
                )
            else:
                d4_pair_value(low_pair, block, 0, phase)
                vec_select(
                    low_mix,
                    bit0,
                    low_pair,
                    low_mix,
                    priority_boost=lookup_priority_boost,
                )
            vec_select(
                work_mix,
                bit1,
                low_mix,
                high_tmp,
                priority_boost=lookup_priority_boost,
            )

            if block in scalar_bit2_depth4_blocks:
                alu_lanes_scalar(
                    ">>",
                    bit0,
                    idxs[block],
                    two_s,
                    priority_boost=lookup_priority_boost,
                )
            else:
                vec_op(
                    ">>",
                    bit0,
                    idxs[block],
                    two_v,
                    priority_boost=lookup_priority_boost,
                )
            vec_select(
                tmp1s[block],
                bit0,
                work_mix,
                tmp1s[block],
                priority_boost=lookup_priority_boost,
            )
            if spill_val:
                add_task(
                    "load",
                    ("vload", vals[block], store_addrs[block]),
                    reads=(store_addrs[block],),
                    writes=range(vals[block], vals[block] + VLEN),
                    extra_deps=(spill_store_task,),
                )

        def round_depth3(ids, round_i, use_scalar_and):
            select_final_depth4 = round_i == rounds - 2
            select_early_depth4 = round_i == 3
            emit_hash(ids, round_i=round_i)
            for block in ids:
                emit_parity(tmp0s[block], vals[block], use_scalar_and)
                if select_final_depth4 and block in final_depth4_select_blocks:
                    precompute_depth4_value(block, "final")
                elif select_early_depth4 and block in early_depth4_select_blocks:
                    precompute_depth4_value(block, "early")
                    vec_op("-", idxs[block], depth3_mask_v, idxs[block])
                    vec_madd(idxs[block], idxs[block], two_v, depth4_base_v)
                    vec_op("+", idxs[block], idxs[block], tmp0s[block])
                else:
                    vec_op("-", idxs[block], depth3_mask_v, idxs[block])
                    vec_madd(idxs[block], idxs[block], two_v, depth4_base_v)
                    vec_op("+", idxs[block], idxs[block], tmp0s[block])

        def round_gather(ids, round_i, update_idx, use_scalar_and):
            for block in ids:
                if (
                    round_i == rounds - 1
                    and block in final_depth4_select_blocks
                ) or (round_i == 4 and block in early_depth4_select_blocks):
                    alu_lanes("^", vals[block], vals[block], tmp1s[block])
                else:
                    for lane in range(VLEN):
                        add_task(
                            "load",
                            ("load_offset", tmp0s[block], idxs[block], lane),
                            reads=(idxs[block] + lane,),
                            writes=(tmp0s[block] + lane,),
                            priority_boost=(
                                final_gather_priority_boost
                                if round_i == rounds - 1
                                and block in final_gather_priority_blocks
                                else 0
                            ),
                        )
                    alu_lanes("^", vals[block], vals[block], tmp0s[block])
            final_xor_scalar = (
                c5_root_s
                if round_i == forest_height and round_i + 1 < rounds
                else c5_s
            )
            emit_hash(ids, final_xor_scalar, round_i=round_i)
            if update_idx:
                for block in ids:
                    emit_parity(tmp0s[block], vals[block], use_scalar_and)
                    vec_select(tmp1s[block], tmp0s[block], add_odd_v, add_even_v)
                    vec_madd(idxs[block], idxs[block], two_v, tmp1s[block])

        def emit_round(ids, round_i):
            depth = round_i if round_i <= forest_height else round_i - (forest_height + 1)
            use_scalar_and = depth > 3
            if depth == 0:
                round_root(
                    ids,
                    round_i=round_i,
                    use_scalar_and=use_scalar_and,
                    root_already_xored=(round_i == forest_height + 1),
                    use_valu_node_xor=False,
                )
            elif depth == 1:
                round_depth1(
                    ids,
                    round_i,
                    use_scalar_and,
                    use_valu_node_xor=False,
                )
            elif depth == 2:
                round_depth2(
                    ids,
                    round_i,
                    use_scalar_and,
                    use_valu_node_xor=False,
                )
            elif depth == 3:
                round_depth3(ids, round_i, use_scalar_and)
            else:
                round_gather(
                    ids,
                    round_i=round_i,
                    update_idx=(round_i != forest_height and round_i != rounds - 1),
                    use_scalar_and=use_scalar_and,
                )

        group_offsets = [0, 0, 6, 8, 8, 11, 13, 17, 15, 19, 20, 12, 22]
        if os.environ.get("GROUP_OFFSETS"):
            group_offsets = [int(x) for x in os.environ["GROUP_OFFSETS"].split(",")]
            assert len(group_offsets) == n_groups and min(group_offsets) >= 0
        for schedule_round in range(rounds + max(group_offsets)):
            for group, ids in enumerate(group_ids):
                round_i = schedule_round - group_offsets[group]
                if 0 <= round_i < rounds:
                    emit_round(ids, round_i)

        for block in tile_ids:
            store_addr = store_addrs[block]
            if compact_tail_store_addrs and block >= 28:
                store_addr = idxs[block]
                load_scalar_const(store_addr, inp_values_p + block * VLEN)
            add_task(
                "store",
                ("vstore", store_addr, vals[block]),
                reads=(store_addr,) + tuple(range(vals[block], vals[block] + VLEN)),
            )

        self.instrs.extend(self.build_scheduled(tasks))
        assert "flow" not in self.instrs[0] and "flow" not in self.instrs[-1]
        self.instrs[0]["flow"] = [("pause",)]
        self.instrs[-1]["flow"] = [("pause",)]

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
