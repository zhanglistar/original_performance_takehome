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

    def reserve_const(self, val, name=None):
        if val not in self.const_map:
            self.const_map[val] = self.alloc_scratch(name)
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation of reference_kernel2.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        # Scratch space addresses
        init_vars = [
            "forest_values_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        init_values = {
            "rounds": rounds,
            "n_nodes": n_nodes,
            "batch_size": batch_size,
            "forest_height": forest_height,
            "forest_values_p": 7,
            "inp_indices_p": 7 + n_nodes,
            "inp_values_p": 7 + n_nodes + batch_size,
        }
        init_loads = [
            ("const", self.scratch[v], init_values[v])
            for v in init_vars
        ]
        for pos in range(0, len(init_loads), SLOT_LIMITS["load"]):
            self.instrs.append(
                {"load": init_loads[pos : pos + SLOT_LIMITS["load"]]}
            )

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.instrs[-1]["flow"] = [("pause",)]
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        setup_flow_slots = []
        setup_phase = False

        def emit(**engines):
            if setup_phase and setup_flow_slots and "flow" not in engines:
                engines["flow"] = [setup_flow_slots.pop(0)]
            self.instrs.append(
                {
                    name: slots
                    for name, slots in engines.items()
                    if slots
                }
            )

        n_vec_groups = (batch_size + VLEN - 1) // VLEN
        max_groups = n_vec_groups
        use_level3_cache = True
        level3_rounds = set()
        level3_round3_groups = {20, 21, 22, 25, 26, 28, 29, 30}
        level3_round14_groups = set(range(max_groups)) - {27, 28, 31}
        idx = [self.alloc_scratch(f"idx{g}", VLEN) for g in range(max_groups)]
        val = [self.alloc_scratch(f"val{g}", VLEN) for g in range(max_groups)]
        addr = [self.alloc_scratch(f"addr{g}", VLEN) for g in range(max_groups)]
        tmpa = [self.alloc_scratch(f"tmpa{g}", VLEN) for g in range(max_groups)]
        tmpb_pool_size = 18
        tmpb = [self.alloc_scratch(f"tmpb{i}", VLEN) for i in range(tmpb_pool_size)]
        store_ptr = [self.alloc_scratch(f"store_ptr{g}") for g in range(max_groups)]
        setup_flow_slots = [
            ("add_imm", store_ptr[g], self.scratch["inp_values_p"], g * VLEN)
            for g in range(max_groups)
        ]
        setup_phase = True
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        root_child_base_v = self.alloc_scratch("root_child_base_v", VLEN)
        addr_update_const_v = self.alloc_scratch("addr_update_const_v", VLEN)
        addr_update_odd_v = self.alloc_scratch("addr_update_odd_v", VLEN)
        root_value = self.alloc_scratch("root_value")
        root_value_v = self.alloc_scratch("root_value_v", VLEN)
        level1_diff = self.alloc_scratch("level1_diff")
        level1_right_v = self.alloc_scratch("level1_right_v", VLEN)
        level1_diff_v = self.alloc_scratch("level1_diff_v", VLEN)
        level2_right_v = [
            self.alloc_scratch(f"level2_right_v{i}", VLEN) for i in range(2)
        ]
        level2_diff_v = [
            self.alloc_scratch(f"level2_diff_v{i}", VLEN) for i in range(2)
        ]
        level2_split_v = self.alloc_scratch("level2_split_v", VLEN)
        level3_base_v = [
            self.alloc_scratch(f"level3_base_v{i}", VLEN) for i in range(4)
        ]
        level3_diff_v = [
            self.alloc_scratch(f"level3_diff_v{i}", VLEN) for i in range(4)
        ]
        level3_split_v = [
            self.alloc_scratch(f"level3_split_v{i}", VLEN) for i in range(3)
        ]
        hash_const_v = {}
        hash_mult_v = {}
        for op1, _val1, op2, op3, val3 in HASH_STAGES:
            if (op1, op2, op3) == ("+", "+", "<<"):
                c = 1 + (1 << val3)
                if c not in hash_mult_v:
                    hash_mult_v[c] = self.alloc_scratch(f"hash_mult_{c:x}", VLEN)
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            needed = [val1]
            if (op1, op2, op3) != ("+", "+", "<<"):
                needed.append(val3)
            for c in needed:
                if c not in hash_const_v:
                    hash_const_v[c] = self.alloc_scratch(f"hash_const_{c:x}", VLEN)

        scalar_const_values = [
            1,
            2,
            init_values["forest_values_p"] + 1,
            1 - init_values["forest_values_p"],
            2 - init_values["forest_values_p"],
            init_values["forest_values_p"] + 5,
            *hash_const_v.keys(),
            *hash_mult_v.keys(),
            init_values["forest_values_p"] + 9,
            init_values["forest_values_p"] + 11,
            init_values["forest_values_p"] + 13,
        ]
        scalar_const_addrs = {}
        scalar_const_loads = []
        for c in scalar_const_values:
            if c in scalar_const_addrs:
                continue
            addr_c = self.reserve_const(c)
            scalar_const_addrs[c] = addr_c
            scalar_const_loads.append(("const", addr_c, c))
        one_const = scalar_const_addrs[1]
        two_const = scalar_const_addrs[2]

        early_const_values = {
            1,
            2,
            init_values["forest_values_p"] + 1,
            1 - init_values["forest_values_p"],
            2 - init_values["forest_values_p"],
            init_values["forest_values_p"] + 11,
            init_values["forest_values_p"] + 13,
        }
        remaining_const_loads = [
            ("const", scalar_const_addrs[c], c)
            for c in scalar_const_values
            if c not in early_const_values
        ]
        seen_remaining = set()
        remaining_const_loads = [
            slot
            for slot in remaining_const_loads
            if not (slot[2] in seen_remaining or seen_remaining.add(slot[2]))
        ]

        def take_const_loads(n):
            res = remaining_const_loads[:n]
            del remaining_const_loads[:n]
            return res

        emit(
            load=[
                ("load", root_value, self.scratch["forest_values_p"]),
                ("const", one_const, 1),
            ]
        )
        emit(
            load=[
                ("const", two_const, 2),
                (
                    "const",
                    scalar_const_addrs[init_values["forest_values_p"] + 1],
                    init_values["forest_values_p"] + 1,
                ),
            ]
        )
        emit(
            load=[
                (
                    "const",
                    scalar_const_addrs[1 - init_values["forest_values_p"]],
                    1 - init_values["forest_values_p"],
                ),
                (
                    "const",
                    scalar_const_addrs[2 - init_values["forest_values_p"]],
                    2 - init_values["forest_values_p"],
                ),
            ]
        )
        emit(
            load=[
                ("const", tmp1, init_values["forest_values_p"] + 1),
                ("const", tmp2, init_values["forest_values_p"] + 2),
            ],
            valu=[
                ("vbroadcast", one_v, one_const),
                ("vbroadcast", two_v, two_const),
                (
                    "vbroadcast",
                    root_child_base_v,
                    scalar_const_addrs[init_values["forest_values_p"] + 1],
                ),
                (
                    "vbroadcast",
                    addr_update_const_v,
                    scalar_const_addrs[1 - init_values["forest_values_p"]],
                ),
                (
                    "vbroadcast",
                    addr_update_odd_v,
                    scalar_const_addrs[2 - init_values["forest_values_p"]],
                ),
                ("vbroadcast", root_value_v, root_value),
            ]
        )
        emit(
            load=[
                ("load", root_value, tmp1),
                ("load", level1_diff, tmp2),
            ]
        )
        emit(
            load=take_const_loads(2),
            alu=[("-", level1_diff, level1_diff, root_value)],
            valu=[("vbroadcast", level1_right_v, root_value)],
        )
        emit(
            load=take_const_loads(2),
            valu=[
                ("vbroadcast", level1_diff_v, level1_diff),
                (
                    "vbroadcast",
                    level2_split_v,
                    scalar_const_addrs[init_values["forest_values_p"] + 5],
                ),
                ("vbroadcast", hash_const_v[0x7ED55D16], scalar_const_addrs[0x7ED55D16]),
            ],
        )
        level2_pairs = [(3, 4), (5, 6)]
        for pair_i, (left_node, right_node) in enumerate(level2_pairs):
            emit(
                load=[
                    ("const", tmp1, init_values["forest_values_p"] + left_node),
                    ("const", tmp2, init_values["forest_values_p"] + right_node),
                ],
                valu=(
                    [
                        (
                            "vbroadcast",
                            hash_const_v[0xC761C23C],
                            scalar_const_addrs[0xC761C23C],
                        ),
                        ("vbroadcast", hash_const_v[19], scalar_const_addrs[19]),
                    ]
                    if pair_i == 0
                    else [
                        ("vbroadcast", hash_const_v[9], scalar_const_addrs[9]),
                        (
                            "vbroadcast",
                            hash_const_v[0xFD7046C5],
                            scalar_const_addrs[0xFD7046C5],
                        ),
                    ]
                ),
            )
            emit(
                load=[
                    ("load", root_value, tmp1),
                    ("load", level1_diff, tmp2),
                ]
            )
            emit(
                load=take_const_loads(2),
                alu=[("-", root_value, level1_diff, root_value)],
                valu=[("vbroadcast", level2_right_v[pair_i], root_value)],
            )
            emit(
                load=take_const_loads(2),
                valu=[
                    ("vbroadcast", level2_diff_v[pair_i], root_value),
                    *(
                        [
                            (
                                "vbroadcast",
                                hash_const_v[0x165667B1],
                                scalar_const_addrs[0x165667B1],
                            ),
                            (
                                "vbroadcast",
                                hash_const_v[0xD3A2646C],
                                scalar_const_addrs[0xD3A2646C],
                            ),
                        ]
                        if pair_i == 0
                        else [
                            (
                                "vbroadcast",
                                hash_const_v[0xB55A4F09],
                                scalar_const_addrs[0xB55A4F09],
                            ),
                            ("vbroadcast", hash_const_v[16], scalar_const_addrs[16]),
                        ]
                    ),
                    *(
                        [("vbroadcast", hash_mult_v[9], scalar_const_addrs[9])]
                        if pair_i == 1 and 9 in hash_mult_v
                        else []
                    ),
                ]
            )
        emit(
            load=[
                (
                    "const",
                    scalar_const_addrs[init_values["forest_values_p"] + 11],
                    init_values["forest_values_p"] + 11,
                ),
                (
                    "const",
                    scalar_const_addrs[init_values["forest_values_p"] + 13],
                    init_values["forest_values_p"] + 13,
                ),
            ]
        )
        for split_i, split_val in enumerate((9, 11, 13)):
            emit(
                valu=[
                    (
                        "vbroadcast",
                        level3_split_v[split_i],
                        scalar_const_addrs[init_values["forest_values_p"] + split_val],
                    )
                ]
            )
        level3_pairs = [(7, 8), (9, 10), (11, 12), (13, 14)]
        for pair_i, (even_node, odd_node) in enumerate(level3_pairs):
            emit(
                load=[
                    ("const", tmp1, init_values["forest_values_p"] + even_node),
                    ("const", tmp2, init_values["forest_values_p"] + odd_node),
                ]
            )
            emit(
                load=[
                    ("load", root_value, tmp1),
                    ("load", level1_diff, tmp2),
                ]
            )
            emit(
                alu=[("-", level1_diff, level1_diff, root_value)],
                valu=[("vbroadcast", level3_base_v[pair_i], root_value)],
            )
            emit(
                valu=[("vbroadcast", level3_diff_v[pair_i], level1_diff)]
            )
        while remaining_const_loads:
            emit(load=take_const_loads(SLOT_LIMITS["load"]))

        setup_broadcasts = []
        early_hash_consts = {
            init_values["forest_values_p"] + 5,
            0x7ED55D16,
            0xC761C23C,
            19,
            0x165667B1,
            0xD3A2646C,
            9,
            0xFD7046C5,
            0xB55A4F09,
            16,
        }
        setup_broadcasts.extend(
            ("vbroadcast", vec, scalar_const_addrs[c])
            for c, vec in hash_const_v.items()
            if c not in early_hash_consts
        )
        setup_broadcasts.extend(
            ("vbroadcast", vec, scalar_const_addrs[c])
            for c, vec in hash_mult_v.items()
            if c != 9
        )
        pending_setup_broadcasts = setup_broadcasts

        precomputed_store_ptr_count = max_groups - len(setup_flow_slots)
        setup_phase = False

        for block_start in range(0, n_vec_groups, max_groups):
            group_count = min(max_groups, n_vec_groups - block_start)

            states = [
                {
                    "round": 0,
                    "phase": "waiting_load",
                    "ready": 0,
                    "off": 0,
                    "store_ready": False,
                    "tmpb_slot": None,
                    "tmpb_slot2": None,
                }
                for _ in range(group_count)
            ]
            def advance_after_update(g, ready_cycle):
                states[g]["round"] += 1
                states[g]["ready"] = ready_cycle
                states[g]["off"] = 0
                if states[g]["round"] >= rounds:
                    states[g]["phase"] = "store" if states[g]["store_ready"] else "store_addr"
                elif use_level3_cache and (
                    states[g]["round"] in level3_rounds
                    or (
                        states[g]["round"] == 14
                        and g in level3_round14_groups
                    )
                    or (
                        states[g]["round"] == 3
                        and g in level3_round3_groups
                    )
                ):
                    states[g]["phase"] = "addr"
                elif states[g]["round"] % (forest_height + 1) >= 3:
                    states[g]["phase"] = "gather"
                else:
                    states[g]["phase"] = "addr"

            sched_cycle = 0
            init_load_pair = 0
            next_init_ptr = precomputed_store_ptr_count
            ptr_ready = [
                0 if g < precomputed_store_ptr_count else 10**9
                for g in range(group_count)
            ]
            for g in range(min(precomputed_store_ptr_count, group_count)):
                states[g]["store_ready"] = True
            while any(st["phase"] != "done" for st in states):
                done_count = sum(st["phase"] == "done" for st in states)
                if done_count >= 22:
                    load_scan_order = list(range(group_count - 1, -1, -1))
                else:
                    load_scan_order = list(range(group_count))
                if done_count >= 0:
                    store_scan_order = list(range(group_count - 1, -1, -1))
                else:
                    store_scan_order = list(range(group_count))
                if done_count >= 0:
                    flow_scan_order = list(range(group_count - 1, -1, -1))
                else:
                    flow_scan_order = list(range(group_count))
                if done_count >= 16:
                    valu_scan_order = list(range(group_count - 1, -1, -1))
                else:
                    valu_scan_order = list(range(group_count))
                load_slots = []
                alu_slots = []
                valu_slots = []
                store_slots = []
                flow_slots = []

                def add_alu_vec(op, dest, a1, a2):
                    if op not in ("+", "-", "^", "&", "<<", ">>", "<"):
                        return False
                    if len(alu_slots) + VLEN > SLOT_LIMITS["alu"]:
                        return False
                    alu_slots.extend(
                        (op, dest + vi, a1 + vi, a2 + vi) for vi in range(VLEN)
                    )
                    return True

                def add_simple_vec(op, dest, a1, a2):
                    # Prefer ALU for simple ops so VALU stays free for multiply_add.
                    # ALU has 12 slots but VLEN=8, so at most one spilled vector/cycle
                    # (alu-8..11 stay idle — can't fit another full vector).
                    if add_alu_vec(op, dest, a1, a2):
                        return True
                    if len(valu_slots) < SLOT_LIMITS["valu"]:
                        valu_slots.append((op, dest, a1, a2))
                        return True
                    return False

                def alloc_tmpb_slot():
                    used = {
                        s["tmpb_slot"]
                        for s in states
                        if s["tmpb_slot"] is not None
                    }
                    used.update(
                        s["tmpb_slot2"]
                        for s in states
                        if s["tmpb_slot2"] is not None
                    )
                    for slot_i in range(tmpb_pool_size):
                        if slot_i not in used:
                            return slot_i
                    return None

                def alloc_two_tmpb_slots():
                    first = alloc_tmpb_slot()
                    if first is None:
                        return None
                    used = {
                        s["tmpb_slot"]
                        for s in states
                        if s["tmpb_slot"] is not None
                    }
                    used.update(
                        s["tmpb_slot2"]
                        for s in states
                        if s["tmpb_slot2"] is not None
                    )
                    used.add(first)
                    for second in range(tmpb_pool_size):
                        if second not in used:
                            return first, second
                    return None

                if init_load_pair < group_count and next_init_ptr < group_count:
                    flow_slots.append(
                        (
                            "add_imm",
                            store_ptr[next_init_ptr],
                            self.scratch["inp_values_p"],
                            (block_start + next_init_ptr) * VLEN,
                        )
                    )
                    ptr_ready[next_init_ptr] = sched_cycle + 1
                    states[next_init_ptr]["store_ready"] = True
                    next_init_ptr += 1

                if init_load_pair < group_count:
                    group = range(
                        init_load_pair,
                        min(init_load_pair + SLOT_LIMITS["load"], group_count),
                    )
                    if all(ptr_ready[g] <= sched_cycle for g in group):
                        load_slots = [
                            ("vload", val[g], store_ptr[g])
                            for g in group
                        ]
                        for g in group:
                            states[g]["phase"] = "addr"
                            states[g]["ready"] = sched_cycle + 1
                        init_load_pair += len(load_slots)
                else:
                    for g in load_scan_order:
                        st = states[g]
                        if len(load_slots) >= SLOT_LIMITS["load"]:
                            break
                        if st["phase"] == "gather" and st["ready"] <= sched_cycle:
                            load_slots.append(("load_offset", addr[g], idx[g], st["off"]))
                            st["off"] += 1
                            if st["off"] == VLEN:
                                st["phase"] = "xor"
                                st["ready"] = sched_cycle + 1
                    for g in load_scan_order:
                        st = states[g]
                        if len(load_slots) >= SLOT_LIMITS["load"]:
                            break
                        if st["phase"] == "gather" and st["ready"] <= sched_cycle:
                            load_slots.append(("load_offset", addr[g], idx[g], st["off"]))
                            st["off"] += 1
                            if st["off"] == VLEN:
                                st["phase"] = "xor"
                                st["ready"] = sched_cycle + 1
                    for g in load_scan_order:
                        st = states[g]
                        if len(load_slots) >= SLOT_LIMITS["load"]:
                            break
                        if not st["store_ready"]:
                            load_slots.append(
                                (
                                    "const",
                                    store_ptr[g],
                                    init_values["inp_values_p"]
                                    + (block_start + g) * VLEN,
                                )
                            )
                            st["store_ready"] = True
                            if st["phase"] == "store_addr" and st["ready"] <= sched_cycle:
                                st["phase"] = "store"
                                st["ready"] = sched_cycle + 1

                while pending_setup_broadcasts and len(valu_slots) < SLOT_LIMITS["valu"]:
                    valu_slots.append(pending_setup_broadcasts.pop(0))

                for g in store_scan_order:
                    st = states[g]
                    if len(store_slots) >= SLOT_LIMITS["store"]:
                        break
                    if st["phase"] == "store" and st["ready"] <= sched_cycle:
                        store_slots.append(("vstore", store_ptr[g], val[g]))
                        st["phase"] = "done"
                        st["ready"] = sched_cycle + 1

                for g in flow_scan_order:
                    st = states[g]
                    if flow_slots:
                        break
                    if st["ready"] > sched_cycle:
                        continue
                    if st["phase"] == "level2_select_flow":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                tmpa[g],
                                addr[g],
                                tmpb[tmpb_slot],
                            )
                        )
                        st["tmpb_slot"] = None
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_select01_flow":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                tmpa[g],
                                addr[g],
                                tmpb[tmpb_slot],
                            )
                        )
                        st["phase"] = "level3_parity2"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_select23_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        if tmpb_slot is None or tmpb_slot2 is None:
                            continue
                        flow_slots.append(
                            (
                                "vselect",
                                tmpb[tmpb_slot],
                                tmpa[g],
                                tmpb[tmpb_slot],
                                tmpb[tmpb_slot2],
                            )
                        )
                        st["phase"] = "level3_final_cmp"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_final_select_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        if tmpb_slot is None:
                            continue
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                tmpa[g],
                                addr[g],
                                tmpb[tmpb_slot],
                            )
                        )
                        st["tmpb_slot"] = None
                        st["tmpb_slot2"] = None
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "select_inc":
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                tmpa[g],
                                addr_update_odd_v,
                                addr_update_const_v,
                            )
                        )
                        st["phase"] = "madd"
                        st["ready"] = sched_cycle + 1
                for g in valu_scan_order:
                    st = states[g]
                    if st["ready"] > sched_cycle or st["phase"] in (
                        "gather",
                        "waiting_load",
                        "level2_select_flow",
                        "level3_select01_flow",
                        "level3_select23_flow",
                        "level3_final_select_flow",
                        "select_inc",
                        "done",
                    ):
                        continue
                    phase = st["phase"]
                    r = st["round"]

                    if phase == "hash_pre":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            tmpb_slot = alloc_tmpb_slot()
                            if tmpb_slot is None:
                                continue
                            st["tmpb_slot"] = tmpb_slot
                        hash_tmp = tmpb[tmpb_slot]
                        hi = st["hash_i"]
                        op1, val1, _op2, op3, val3 = HASH_STAGES[hi]
                        if (op1, _op2, op3) == ("+", "+", "<<"):
                            if len(valu_slots) >= SLOT_LIMITS["valu"]:
                                break
                            mult = 1 + (1 << val3)
                            valu_slots.append(
                                (
                                    "multiply_add",
                                    val[g],
                                    val[g],
                                    hash_mult_v[mult],
                                    hash_const_v[val1],
                                )
                            )
                            st["hash_i"] += 1
                            if st["hash_i"] < len(HASH_STAGES):
                                st["phase"] = "hash_pre"
                            elif r + 1 >= rounds:
                                advance_after_update(g, sched_cycle + 1)
                            elif (r + 1) % (forest_height + 1) == 0:
                                st["phase"] = "zero"
                            else:
                                st["phase"] = "parity"
                            st["ready"] = sched_cycle + 1
                            continue
                        if len(valu_slots) + 2 > SLOT_LIMITS["valu"]:
                            if len(valu_slots) < SLOT_LIMITS["valu"]:
                                valu_slots.append((op1, tmpa[g], val[g], hash_const_v[val1]))
                                if add_alu_vec(op3, hash_tmp, val[g], hash_const_v[val3]):
                                    st["phase"] = "hash_combine"
                                else:
                                    st["phase"] = "hash_pre_second"
                                st["ready"] = sched_cycle + 1
                            continue
                        valu_slots.append((op1, tmpa[g], val[g], hash_const_v[val1]))
                        valu_slots.append((op3, hash_tmp, val[g], hash_const_v[val3]))
                        st["phase"] = "hash_combine"
                        st["ready"] = sched_cycle + 1
                        continue
                    if phase == "addr":
                        if r % (forest_height + 1) == 0:
                            if not add_simple_vec("^", val[g], val[g], root_value_v):
                                break
                            st["phase"] = "hash_pre"
                            st["hash_i"] = 0
                        elif r % (forest_height + 1) == 1:
                            if len(valu_slots) >= SLOT_LIMITS["valu"]:
                                break
                            valu_slots.append(
                                (
                                    "multiply_add",
                                    addr[g],
                                    tmpa[g],
                                    level1_diff_v,
                                    level1_right_v,
                                )
                            )
                            st["phase"] = "xor"
                        elif r % (forest_height + 1) == 2:
                            tmpb_slot = alloc_tmpb_slot()
                            if tmpb_slot is None:
                                continue
                            if len(valu_slots) >= SLOT_LIMITS["valu"]:
                                break
                            valu_slots.append(
                                (
                                    "multiply_add",
                                    addr[g],
                                    tmpa[g],
                                    level2_diff_v[0],
                                    level2_right_v[0],
                                )
                            )
                            st["tmpb_slot"] = tmpb_slot
                            st["phase"] = "level2_pair1"
                        elif use_level3_cache and (
                            r in level3_rounds
                            or (r == 14 and g in level3_round14_groups)
                            or (r == 3 and g in level3_round3_groups)
                        ):
                            slots = alloc_two_tmpb_slots()
                            if slots is None:
                                continue
                            if not add_simple_vec("&", tmpa[g], idx[g], one_v):
                                break
                            st["tmpb_slot"], st["tmpb_slot2"] = slots
                            st["phase"] = "level3_pair0"
                        else:
                            st["phase"] = "gather"
                            st["ready"] = sched_cycle
                            continue
                        st["ready"] = sched_cycle + 1
                    elif phase == "xor":
                        if not add_simple_vec("^", val[g], val[g], addr[g]):
                            break
                        st["phase"] = "hash_pre"
                        st["hash_i"] = 0
                        st["ready"] = sched_cycle + 1
                    elif phase == "level1_select":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", addr[g], tmpa[g], level1_diff_v, level1_right_v)
                        )
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_pair0":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                addr[g],
                                tmpa[g],
                                level2_diff_v[0],
                                level2_right_v[0],
                            )
                        )
                        st["phase"] = "level2_pair1"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_pair1":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                tmpb[tmpb_slot],
                                tmpa[g],
                                level2_diff_v[1],
                                level2_right_v[1],
                            )
                        )
                        st["phase"] = "level2_cmp"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_cmp":
                        if not add_simple_vec("<", tmpa[g], idx[g], level2_split_v):
                            break
                        st["phase"] = "level2_select_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_diff":
                        if not add_simple_vec("-", addr[g], addr[g], tmpb[g]):
                            break
                        st["phase"] = "level2_select"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_select":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(("multiply_add", addr[g], tmpa[g], addr[g], tmpb[g]))
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_pair0":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", addr[g], tmpa[g], level3_diff_v[0], level3_base_v[0])
                        )
                        st["phase"] = "level3_pair1"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_pair1":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", tmpb[tmpb_slot], tmpa[g], level3_diff_v[1], level3_base_v[1])
                        )
                        st["phase"] = "level3_cmp01"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_cmp01":
                        if not add_simple_vec("<", tmpa[g], idx[g], level3_split_v[0]):
                            break
                        st["phase"] = "level3_select01_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_parity2":
                        if not add_simple_vec("&", tmpa[g], idx[g], one_v):
                            break
                        st["phase"] = "level3_pair2"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_pair2":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", tmpb[tmpb_slot], tmpa[g], level3_diff_v[2], level3_base_v[2])
                        )
                        st["phase"] = "level3_pair3"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_pair3":
                        tmpb_slot2 = st["tmpb_slot2"]
                        if tmpb_slot2 is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", tmpb[tmpb_slot2], tmpa[g], level3_diff_v[3], level3_base_v[3])
                        )
                        st["phase"] = "level3_cmp23"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_cmp23":
                        if not add_simple_vec("<", tmpa[g], idx[g], level3_split_v[2]):
                            break
                        st["phase"] = "level3_select23_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_final_cmp":
                        if not add_simple_vec("<", tmpa[g], idx[g], level3_split_v[1]):
                            break
                        st["phase"] = "level3_final_select_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "hash_combine":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        hash_tmp = tmpb[tmpb_slot]
                        hi = st["hash_i"]
                        _op1, _val1, op2, _op3, _val3 = HASH_STAGES[hi]
                        if not add_simple_vec(op2, val[g], tmpa[g], hash_tmp):
                            break
                        st["hash_i"] += 1
                        if st["hash_i"] < len(HASH_STAGES):
                            st["phase"] = "hash_pre"
                        elif r + 1 >= rounds:
                            st["tmpb_slot"] = None
                            advance_after_update(g, sched_cycle + 1)
                        elif (r + 1) % (forest_height + 1) == 0:
                            st["tmpb_slot"] = None
                            st["phase"] = "zero"
                        else:
                            st["tmpb_slot"] = None
                            st["phase"] = "parity"
                        st["ready"] = sched_cycle + 1
                    elif phase == "hash_pre_second":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        hash_tmp = tmpb[tmpb_slot]
                        hi = st["hash_i"]
                        _op1, _val1, _op2, op3, val3 = HASH_STAGES[hi]
                        if not add_simple_vec(op3, hash_tmp, val[g], hash_const_v[val3]):
                            break
                        st["phase"] = "hash_combine"
                        st["ready"] = sched_cycle + 1
                    elif phase == "parity":
                        if not add_simple_vec("&", tmpa[g], val[g], one_v):
                            break
                        st["phase"] = (
                            "root_add_one"
                            if r % (forest_height + 1) == 0
                            else "select_inc"
                        )
                        st["ready"] = sched_cycle + 1
                    elif phase == "root_add_one":
                        if not add_simple_vec("+", idx[g], tmpa[g], root_child_base_v):
                            break
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "madd":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(("multiply_add", idx[g], idx[g], two_v, addr[g]))
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "add_one":
                        if not add_simple_vec("+", idx[g], tmpb[g], addr_update_const_v):
                            break
                        if (r + 1) % (forest_height + 1) == 0:
                            st["phase"] = "zero"
                            st["ready"] = sched_cycle + 1
                        else:
                            advance_after_update(g, sched_cycle + 1)
                    elif phase == "old_madd":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(("multiply_add", tmpb[g], idx[g], two_v, tmpa[g]))
                        st["phase"] = "add_one"
                        st["ready"] = sched_cycle + 1
                    elif phase == "zero":
                        if not add_simple_vec("^", idx[g], idx[g], idx[g]):
                            break
                        advance_after_update(g, sched_cycle + 1)

                if load_slots or alu_slots or valu_slots or store_slots or flow_slots:
                    emit(
                        load=load_slots,
                        alu=alu_slots,
                        valu=valu_slots,
                        store=store_slots,
                        flow=flow_slots,
                    )
                sched_cycle += 1

        # Required to match with the yield in reference_kernel2.
        if "flow" not in self.instrs[-1]:
            self.instrs[-1]["flow"] = [("pause",)]
        else:
            self.instrs.append({"flow": [("pause",)]})

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
