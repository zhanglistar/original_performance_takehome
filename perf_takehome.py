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
        dep_count = [0] * len(tasks)
        for i, task in enumerate(tasks):
            dep_count[i] = len(task["deps"])
            for dep in task["deps"]:
                dependents[dep].append(i)

        priority = [0] * len(tasks)
        downstream = [0] * len(tasks)
        for i in range(len(tasks) - 1, -1, -1):
            priority[i] = 1 + max((priority[j] for j in dependents[i]), default=0)
            downstream[i] = min(
                100000, len(dependents[i]) + sum(downstream[j] for j in dependents[i])
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
        while remaining:
            bundle = {}
            selected = []
            for engine in engine_order:
                limit = SLOT_LIMITS[engine]
                candidates = [
                    i for i in ready if not scheduled[i] and tasks[i]["engine"] == engine
                ]
                if tie_mode == "early":
                    candidates.sort(key=lambda i: (-priority[i], i))
                elif tie_mode == "fanout":
                    candidates.sort(key=lambda i: (-priority[i], -downstream[i], -i))
                elif tie_mode == "low_pressure":
                    candidates.sort(
                        key=lambda i: (-priority[i], len(tasks[i].get("anti_deps", ())), -i)
                    )
                else:
                    candidates.sort(key=lambda i: (-priority[i], -i))
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
                        bundle.setdefault(engine, []).append(tasks[i]["slot"])
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

        def add_task(engine, slot, reads=(), writes=()):
            reads = tuple(reads)
            writes = tuple(writes)
            deps = set()
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
            tasks.append(
                {
                    "engine": engine,
                    "slot": slot,
                    "deps": deps,
                    "anti_deps": anti_deps,
                }
            )

            write_set = set(writes)
            for addr in reads:
                if addr not in write_set:
                    last_readers[addr].add(task_id)
            for addr in writes:
                last_readers[addr].clear()
                last_writer[addr] = task_id
            return task_id

        def vec_op(op, dest, a, b):
            add_task(
                "valu",
                (op, dest, a, b),
                reads=tuple(range(a, a + VLEN)) + tuple(range(b, b + VLEN)),
                writes=range(dest, dest + VLEN),
            )

        def vec_madd(dest, a, b, c):
            add_task(
                "valu",
                ("multiply_add", dest, a, b, c),
                reads=(
                    tuple(range(a, a + VLEN))
                    + tuple(range(b, b + VLEN))
                    + tuple(range(c, c + VLEN))
                ),
                writes=range(dest, dest + VLEN),
            )

        def vec_select(dest, cond, a, b):
            add_task(
                "flow",
                ("vselect", dest, cond, a, b),
                reads=(
                    tuple(range(cond, cond + VLEN))
                    + tuple(range(a, a + VLEN))
                    + tuple(range(b, b + VLEN))
                ),
                writes=range(dest, dest + VLEN),
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

        def alu_lanes_scalar(op, dest, a, b_scalar):
            for lane in range(VLEN):
                add_task(
                    "alu",
                    (op, dest + lane, a + lane, b_scalar),
                    reads=(a + lane, b_scalar),
                    writes=(dest + lane,),
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
        d4_flow_pairs = set()
        for i in range(8):
            if i not in d4_flow_pairs:
                vec_op("-", d4_odd_or_diff[i], d4_odd_or_diff[i], d4_even[i])

        bit0 = alloc_vec("bit0")
        bit1 = alloc_vec("bit1")
        mix = alloc_vec("mix")
        pair = alloc_vec("pair")
        high_tmp = alloc_vec("high_tmp")

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
            base = self.alloc_scratch(f"value_base_{block}")
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

        valu_c5_xor_blocks = {30}

        def emit_hash(ids, final_xor_scalar=c5_s, round_i=None):
            for block in ids:
                vec_madd(vals[block], vals[block], m4097_v, c0_v)

            for block in ids:
                vec_op(">>", tmp0s[block], vals[block], sh19_v)
                alu_lanes_scalar("^", vals[block], vals[block], c1_s)
            for block in ids:
                vec_op("^", vals[block], vals[block], tmp0s[block])

            for block in ids:
                vec_madd(tmp0s[block], vals[block], m33_v, c23_v)
                vec_madd(vals[block], vals[block], m16896_v, c2sh9_v)
            for block in ids:
                vec_op("^", vals[block], vals[block], tmp0s[block])

            for block in ids:
                vec_madd(vals[block], vals[block], m9_v, c4_v)

            for block in ids:
                vec_op(">>", tmp0s[block], vals[block], sh16_v)
                if final_xor_scalar is not None:
                    if (
                        final_xor_scalar == c5_s
                        and round_i == rounds - 1
                        and block in valu_c5_xor_blocks
                    ):
                        vec_op("^", vals[block], vals[block], c5_v)
                    else:
                        alu_lanes_scalar("^", vals[block], vals[block], final_xor_scalar)
            for block in ids:
                vec_op("^", vals[block], vals[block], tmp0s[block])

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
        scalar_bit_depth4_blocks = set()

        def d4_pair_value(dest, block, pair_i):
            if pair_i in d4_flow_pairs:
                vec_select(
                    dest,
                    tmp0s[block],
                    d4_odd_or_diff[pair_i],
                    d4_even[pair_i],
                )
            else:
                vec_madd(dest, tmp0s[block], d4_odd_or_diff[pair_i], d4_even[pair_i])

        def precompute_depth4_value(block):
            if block in scalar_bit_depth4_blocks:
                alu_lanes_scalar("&", bit0, idxs[block], one_s)
                alu_lanes_scalar(">>", bit1, idxs[block], one_s)
                alu_lanes_scalar("&", bit1, bit1, one_s)
            else:
                vec_op("&", bit0, idxs[block], one_v)
                vec_op(">>", bit1, idxs[block], one_v)
                vec_op("&", bit1, bit1, one_v)

            d4_pair_value(mix, block, 7)
            d4_pair_value(pair, block, 6)
            vec_select(tmp1s[block], bit0, pair, mix)

            d4_pair_value(mix, block, 5)
            d4_pair_value(pair, block, 4)
            vec_select(mix, bit0, pair, mix)
            vec_select(tmp1s[block], bit1, mix, tmp1s[block])

            d4_pair_value(mix, block, 3)
            d4_pair_value(pair, block, 2)
            vec_select(high_tmp, bit0, pair, mix)

            d4_pair_value(mix, block, 1)
            d4_pair_value(pair, block, 0)
            vec_select(mix, bit0, pair, mix)
            vec_select(mix, bit1, mix, high_tmp)

            if block in scalar_bit_depth4_blocks:
                alu_lanes_scalar(">>", bit0, idxs[block], two_s)
            else:
                vec_op(">>", bit0, idxs[block], two_v)
            vec_select(tmp1s[block], bit0, mix, tmp1s[block])

        def round_depth3(ids, round_i, use_scalar_and):
            select_final_depth4 = round_i == rounds - 2
            select_early_depth4 = round_i == 3
            emit_hash(ids, round_i=round_i)
            for block in ids:
                emit_parity(tmp0s[block], vals[block], use_scalar_and)
                if select_final_depth4 and block in final_depth4_select_blocks:
                    precompute_depth4_value(block)
                elif select_early_depth4 and block in early_depth4_select_blocks:
                    precompute_depth4_value(block)
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
            add_task(
                "store",
                ("vstore", store_addrs[block], vals[block]),
                reads=(store_addrs[block],) + tuple(range(vals[block], vals[block] + VLEN)),
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
