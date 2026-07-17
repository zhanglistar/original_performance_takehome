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

from collections import Counter, defaultdict
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
    max_groups_limit = 32
    use_level4_scalar_cache = True
    level4_scalar_groups = {22}
    use_level3_descriptor_select = False
    preschedule_level3_descriptor = False
    tmpb_pool_size = 13
    prep_flow_yield_stride = 0
    prep_start_hash_stage = 0
    level3_prep_alu_hybrid = False
    use_dedicated_l3_prep_pool = False
    l3_prep_bank_count = 2
    recompute_prepared_parity = True
    preschedule_level3_round14 = True
    use_virtual_tmp_allocator = False
    virtual_tmp_capacity = 13
    virtual_tmp_spill_vectors = 0
    virtual_schedule_chunk_size = 0
    use_additive_index_update = False
    use_dag_scheduler = False
    dag_critical_slack = 0
    dag_virtual_pressure_aware = False
    use_scalar_root_value = False
    strict_virtual_coloring = True
    use_biased_heap_index = False
    biased_gather_addr_alu = False
    reuse_addr_as_tmpa = False
    use_bias_free_c5 = False
    bias_c5_gather_alu = False
    bias_c5_edge_alu = False
    rename_extra_words = 3
    bias_hash5_shift_alu = False
    bias_hash5_combine_alu = False
    bias_keep_initial_unbiased = False
    simple_valu_first_limit = 4
    simple_valu_first_done_threshold = 0
    base_ready_done_threshold = 21
    base_ready_valu_limit = 4
    hash_half_alu_done_threshold = 18
    level3_round3_groups = {2, 20, 21, 22, 25, 26, 29, 30, 31}
    level3_round14_groups = set(range(32)) - {28, 29, 30, 31}
    tail_load_priority = (30, 31, 27, 29, 28)
    double_gather_done_threshold = 20

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.virtual_vector_bases = set()
        self.virtual_tmp_tokens = {}
        self.tmpb_physical_bases = []
        self.virtual_start_delays = {}
        self.virtual_color_failure = None

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

    def slot_rw(self, engine, slot):
        reads = set()
        writes = set()

        def add_read(addr, length=1):
            for i in range(length):
                reads.add(addr + i)

        def add_write(addr, length=1):
            for i in range(length):
                writes.add(addr + i)

        op = slot[0]
        if engine == "alu":
            _, dest, a1, a2 = slot
            add_read(a1)
            add_read(a2)
            add_write(dest)
        elif engine == "valu":
            if op == "vbroadcast":
                _, dest, src = slot
                add_read(src)
                add_write(dest, VLEN)
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                add_read(a, VLEN)
                add_read(b, VLEN)
                add_read(c, VLEN)
                add_write(dest, VLEN)
            else:
                _, dest, a1, a2 = slot
                add_read(a1, VLEN)
                add_read(a2, VLEN)
                add_write(dest, VLEN)
        elif engine == "load":
            if op == "const":
                _, dest, _val = slot
                add_write(dest)
            elif op == "load":
                _, dest, addr = slot
                add_read(addr)
                add_write(dest)
            elif op == "vload":
                _, dest, addr = slot
                add_read(addr)
                add_write(dest, VLEN)
            elif op == "load_offset":
                _, dest, addr, offset = slot
                add_read(addr + offset)
                add_write(dest + offset)
        elif engine == "store":
            if op == "store":
                _, addr, src = slot
                add_read(addr)
                add_read(src)
            elif op == "vstore":
                _, addr, src = slot
                add_read(addr)
                add_read(src, VLEN)
        elif engine == "flow":
            if op == "add_imm":
                _, dest, addr, _imm = slot
                add_read(addr)
                add_write(dest)
            elif op == "select":
                _, dest, cond, a, b = slot
                add_read(cond)
                add_read(a)
                add_read(b)
                add_write(dest)
            elif op == "vselect":
                _, dest, cond, a, b = slot
                add_read(cond, VLEN)
                add_read(a, VLEN)
                add_read(b, VLEN)
                add_write(dest, VLEN)
        return reads, writes

    def schedule_segment(self, instrs):
        ops = []
        for instr in instrs:
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    if engine == "flow" and slot[0] == "pause":
                        continue
                    ops.append((engine, slot))

        reg_avail = defaultdict(int)
        reg_last_read = defaultdict(lambda: -1)
        reg_last_write = defaultdict(lambda: -1)
        usage = defaultdict(lambda: defaultdict(int))
        schedule = defaultdict(list)
        started_virtuals = set()

        for engine, slot in ops:
            reads, writes = self.slot_rw(engine, slot)
            cycle = 0
            for addr in reads:
                cycle = max(cycle, reg_avail[addr])
            for addr in writes:
                cycle = max(
                    cycle,
                    reg_last_read[addr],
                    reg_last_write[addr] + 1,
                )
                if addr >= SCRATCH_SIZE:
                    base = SCRATCH_SIZE + ((addr - SCRATCH_SIZE) // VLEN) * VLEN
                    if base not in started_virtuals:
                        cycle = max(
                            cycle, self.virtual_start_delays.get(base, 0)
                        )
            while usage[cycle][engine] >= SLOT_LIMITS[engine]:
                cycle += 1
            schedule[cycle].append((engine, slot))
            usage[cycle][engine] += 1

            for addr in reads:
                reg_last_read[addr] = max(reg_last_read[addr], cycle)
            for addr in writes:
                reg_avail[addr] = cycle + 1
                reg_last_write[addr] = cycle
                if addr >= SCRATCH_SIZE:
                    base = SCRATCH_SIZE + ((addr - SCRATCH_SIZE) // VLEN) * VLEN
                    started_virtuals.add(base)

        if not schedule:
            return []
        res = []
        for cycle in range(max(schedule) + 1):
            bundle = {}
            for engine, slot in schedule[cycle]:
                bundle.setdefault(engine, []).append(slot)
            res.append(bundle)
        return res

    def schedule_segment_dag(self, instrs):
        ops = []
        for instr in instrs:
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    if engine == "flow" and slot[0] == "pause":
                        continue
                    ops.append((engine, slot))

        if not ops:
            return []

        # Edges carry the minimum cycle distance. RAW/WAW need the producer's
        # result, while WAR may read the old value and overwrite it in one bundle.
        predecessors = [dict() for _ in ops]
        successors = [dict() for _ in ops]
        last_write = {}
        readers_since_write = defaultdict(set)

        def add_edge(src, dst, latency):
            if src is None or src == dst:
                return
            old = predecessors[dst].get(src, -1)
            if latency <= old:
                return
            predecessors[dst][src] = latency
            successors[src][dst] = latency

        for op_i, (engine, slot) in enumerate(ops):
            reads, writes = self.slot_rw(engine, slot)
            for addr in reads:
                add_edge(last_write.get(addr), op_i, 1)
            for addr in writes:
                add_edge(last_write.get(addr), op_i, 1)
                for reader in readers_since_write[addr]:
                    add_edge(reader, op_i, 0)

            for addr in writes:
                last_write[addr] = op_i
                readers_since_write[addr].clear()
            for addr in reads - writes:
                readers_since_write[addr].add(op_i)

        critical = [0] * len(ops)
        for op_i in range(len(ops) - 1, -1, -1):
            if successors[op_i]:
                critical[op_i] = max(
                    latency + critical[succ]
                    for succ, latency in successors[op_i].items()
                )

        virtual_word_base = {
            base + lane: base
            for base in self.virtual_vector_bases
            for lane in range(VLEN)
        }
        virtual_bases_by_op = []
        virtual_touch_remaining = Counter()
        for engine, slot in ops:
            reads, writes = self.slot_rw(engine, slot)
            bases = {
                virtual_word_base[addr]
                for addr in reads | writes
                if addr in virtual_word_base
            }
            virtual_bases_by_op.append(bases)
            virtual_touch_remaining.update(bases)
        active_virtuals = set()
        virtual_capacity = len(self.tmpb_physical_bases)

        remaining = [len(preds) for preds in predecessors]
        earliest = [0] * len(ops)
        ready = set(i for i, count in enumerate(remaining) if count == 0)
        schedule = []
        scheduled_count = 0
        cycle = 0

        while scheduled_count < len(ops):
            bundle = defaultdict(list)
            made_progress = True
            while made_progress:
                made_progress = False
                candidates = [
                    op_i
                    for op_i in ready
                    if earliest[op_i] <= cycle
                    and len(bundle[ops[op_i][0]])
                    < SLOT_LIMITS[ops[op_i][0]]
                    and (
                        not self.dag_virtual_pressure_aware
                        or not (
                            virtual_bases_by_op[op_i] - active_virtuals
                        )
                        or len(active_virtuals) < virtual_capacity
                    )
                ]
                if not candidates:
                    break
                max_critical = max(critical[i] for i in candidates)
                critical_floor = max_critical - self.dag_critical_slack
                priority_candidates = [
                    i for i in candidates if critical[i] >= critical_floor
                ]
                op_i = min(
                    priority_candidates,
                    key=lambda i: (
                        -sum(
                            base in active_virtuals
                            and virtual_touch_remaining[base] == 1
                            for base in virtual_bases_by_op[i]
                        ),
                        i,
                    ),
                )
                ready.remove(op_i)
                engine, slot = ops[op_i]
                bundle[engine].append(slot)
                scheduled_count += 1
                made_progress = True

                for base in virtual_bases_by_op[op_i]:
                    active_virtuals.add(base)
                    virtual_touch_remaining[base] -= 1
                    if virtual_touch_remaining[base] == 0:
                        active_virtuals.remove(base)

                for succ, latency in successors[op_i].items():
                    earliest[succ] = max(earliest[succ], cycle + latency)
                    remaining[succ] -= 1
                    if remaining[succ] == 0:
                        ready.add(succ)

            schedule.append(dict(bundle))
            cycle += 1

        while schedule and not schedule[-1]:
            schedule.pop()
        return schedule

    def color_virtual_tmp_vectors(self, instrs):
        if not self.virtual_vector_bases:
            return instrs

        virtual_word_base = {
            base + lane: base
            for base in self.virtual_vector_bases
            for lane in range(VLEN)
        }
        intervals = {}
        for cycle, instr in enumerate(instrs):
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    reads, writes = self.slot_rw(engine, slot)
                    touched_bases = {
                        virtual_word_base[addr]
                        for addr in reads | writes
                        if addr in virtual_word_base
                    }
                    for base in touched_bases:
                        base_reads = any(
                            virtual_word_base.get(addr) == base for addr in reads
                        )
                        base_writes = any(
                            virtual_word_base.get(addr) == base for addr in writes
                        )
                        if base not in intervals:
                            intervals[base] = {
                                "start": cycle,
                                "end": cycle,
                                "first_reads": base_reads,
                                "first_writes": base_writes,
                                "last_reads": base_reads,
                                "last_writes": base_writes,
                            }
                        else:
                            interval = intervals[base]
                            if cycle == interval["start"]:
                                interval["first_reads"] |= base_reads
                                interval["first_writes"] |= base_writes
                            if cycle > interval["end"]:
                                interval["end"] = cycle
                                interval["last_reads"] = base_reads
                                interval["last_writes"] = base_writes
                            else:
                                interval["last_reads"] |= base_reads
                                interval["last_writes"] |= base_writes

        color_ends = []
        color_last_writes = []
        base_to_color = {}
        for base, interval in sorted(
            intervals.items(),
            key=lambda item: (item[1]["start"], item[1]["end"]),
        ):
            start = interval["start"]
            end = interval["end"]
            color = next(
                (
                    color_i
                    for color_i, color_end in enumerate(color_ends)
                    if color_end < start
                    or (
                        not self.strict_virtual_coloring
                        and
                        color_end == start
                        and not color_last_writes[color_i]
                        and interval["first_writes"]
                        and not interval["first_reads"]
                    )
                ),
                None,
            )
            if color is None:
                color = len(color_ends)
                if color >= len(self.tmpb_physical_bases):
                    self.virtual_color_failure = (base, start, color + 1)
                    raise AssertionError(
                        f"Virtual tmp coloring needs {color + 1} vectors, "
                        f"only {len(self.tmpb_physical_bases)} available"
                    )
                color_ends.append(end)
                color_last_writes.append(interval["last_writes"])
            else:
                color_ends[color] = end
                color_last_writes[color] = interval["last_writes"]
            base_to_color[base] = color

        word_map = {
            base + lane: self.tmpb_physical_bases[color] + lane
            for base, color in base_to_color.items()
            for lane in range(VLEN)
        }

        def mapped(addr):
            return word_map.get(addr, addr)

        def rewrite(engine, slot):
            op = slot[0]
            if engine in ("alu", "valu"):
                if engine == "valu" and op == "vbroadcast":
                    _, dest, src = slot
                    return (op, mapped(dest), mapped(src))
                if engine == "valu" and op == "multiply_add":
                    _, dest, a, b, c = slot
                    return (op, mapped(dest), mapped(a), mapped(b), mapped(c))
                _, dest, a1, a2 = slot
                return (op, mapped(dest), mapped(a1), mapped(a2))
            if engine == "load":
                if op == "const":
                    _, dest, value = slot
                    return (op, mapped(dest), value)
                if op in ("load", "vload"):
                    _, dest, addr = slot
                    return (op, mapped(dest), mapped(addr))
                if op == "load_offset":
                    _, dest, addr, offset = slot
                    return (op, mapped(dest), mapped(addr), offset)
            if engine == "store":
                _, addr, src = slot
                return (op, mapped(addr), mapped(src))
            if engine == "flow":
                if op == "add_imm":
                    _, dest, addr, imm = slot
                    return (op, mapped(dest), mapped(addr), imm)
                if op in ("select", "vselect"):
                    _, dest, cond, a, b = slot
                    return (
                        op,
                        mapped(dest),
                        mapped(cond),
                        mapped(a),
                        mapped(b),
                    )
            return slot

        return [
            {
                engine: [rewrite(engine, slot) for slot in slots]
                for engine, slots in instr.items()
            }
            for instr in instrs
        ]

    def schedule_segment_renamed(self, instrs):
        ops = []
        for instr in instrs:
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    if engine == "flow" and slot[0] == "pause":
                        continue
                    ops.append((engine, slot))

        vector_addrs = set()
        for addr, (_name, length) in self.scratch_debug.items():
            if length == VLEN:
                vector_addrs.update(range(addr, addr + VLEN))

        phys = list(range(SCRATCH_SIZE))
        next_free = self.scratch_ptr

        def mapped_vec_base(addr):
            base = phys[addr]
            for vi in range(VLEN):
                if phys[addr + vi] != base + vi:
                    raise AssertionError("Vector scratch mapping lost contiguity")
            return base

        def logical_rw(engine, slot):
            op = slot[0]
            reads = []
            writes = []
            if engine == "alu":
                _, dest, a1, a2 = slot
                reads = [("s", a1), ("s", a2)]
                writes = [("s", dest)]
            elif engine == "valu":
                if op == "vbroadcast":
                    _, dest, src = slot
                    reads = [("s", src)]
                    writes = [("v", dest)]
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    reads = [("v", a), ("v", b), ("v", c)]
                    writes = [("v", dest)]
                else:
                    _, dest, a1, a2 = slot
                    reads = [("v", a1), ("v", a2)]
                    writes = [("v", dest)]
            elif engine == "load":
                if op == "const":
                    _, dest, _val = slot
                    writes = [("s", dest)]
                elif op == "load":
                    _, dest, addr = slot
                    reads = [("s", addr)]
                    writes = [("s", dest)]
                elif op == "vload":
                    _, dest, addr = slot
                    reads = [("s", addr)]
                    writes = [("v", dest)]
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    reads = [("lane", addr, offset)]
                    writes = [("lane", dest, offset)]
            elif engine == "store":
                if op == "store":
                    _, addr, src = slot
                    reads = [("s", addr), ("s", src)]
                elif op == "vstore":
                    _, addr, src = slot
                    reads = [("s", addr), ("v", src)]
            elif engine == "flow":
                if op == "add_imm":
                    _, dest, addr, _imm = slot
                    reads = [("s", addr)]
                    writes = [("s", dest)]
                elif op == "select":
                    _, dest, cond, a, b = slot
                    reads = [("s", cond), ("s", a), ("s", b)]
                    writes = [("s", dest)]
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    reads = [("v", cond), ("v", a), ("v", b)]
                    writes = [("v", dest)]
            return reads, writes

        def phys_reads(items):
            res = []
            for item in items:
                if item[0] == "s":
                    res.append(phys[item[1]])
                elif item[0] == "v":
                    base = mapped_vec_base(item[1])
                    res.extend(range(base, base + VLEN))
                elif item[0] == "lane":
                    res.append(phys[item[1] + item[2]])
            return res

        def alloc_words(count):
            nonlocal next_free
            rename_limit = min(
                SCRATCH_SIZE, self.scratch_ptr + self.rename_extra_words
            )
            if next_free + count > rename_limit:
                return None
            addr = next_free
            next_free += count
            return addr

        def rewrite_slot(engine, slot):
            op = slot[0]
            if engine == "alu":
                _, dest, a1, a2 = slot
                return (op, phys[dest], phys[a1], phys[a2])
            if engine == "valu":
                if op == "vbroadcast":
                    _, dest, src = slot
                    return (op, mapped_vec_base(dest), phys[src])
                if op == "multiply_add":
                    _, dest, a, b, c = slot
                    return (
                        op,
                        mapped_vec_base(dest),
                        mapped_vec_base(a),
                        mapped_vec_base(b),
                        mapped_vec_base(c),
                    )
                _, dest, a1, a2 = slot
                return (op, mapped_vec_base(dest), mapped_vec_base(a1), mapped_vec_base(a2))
            if engine == "load":
                if op == "const":
                    _, dest, val = slot
                    return (op, phys[dest], val)
                if op == "load":
                    _, dest, addr = slot
                    return (op, phys[dest], phys[addr])
                if op == "vload":
                    _, dest, addr = slot
                    return (op, mapped_vec_base(dest), phys[addr])
                if op == "load_offset":
                    _, dest, addr, offset = slot
                    return (op, mapped_vec_base(dest), mapped_vec_base(addr), offset)
            if engine == "store":
                if op == "store":
                    _, addr, src = slot
                    return (op, phys[addr], phys[src])
                if op == "vstore":
                    _, addr, src = slot
                    return (op, phys[addr], mapped_vec_base(src))
            if engine == "flow":
                if op == "add_imm":
                    _, dest, addr, imm = slot
                    return (op, phys[dest], phys[addr], imm)
                if op == "select":
                    _, dest, cond, a, b = slot
                    return (op, phys[dest], phys[cond], phys[a], phys[b])
                if op == "vselect":
                    _, dest, cond, a, b = slot
                    return (
                        op,
                        mapped_vec_base(dest),
                        mapped_vec_base(cond),
                        mapped_vec_base(a),
                        mapped_vec_base(b),
                    )
            return slot

        reg_avail = defaultdict(int)
        reg_last_read = defaultdict(lambda: -1)
        reg_last_write = defaultdict(lambda: -1)
        usage = defaultdict(lambda: defaultdict(int))
        schedule = defaultdict(list)

        for engine, slot in ops:
            reads, writes = logical_rw(engine, slot)
            cycle = 0
            for addr in phys_reads(reads):
                cycle = max(cycle, reg_avail[addr])

            while True:
                while usage[cycle][engine] >= SLOT_LIMITS[engine]:
                    cycle += 1

                delayed_cycle = cycle
                allocations = []
                for write in writes:
                    if write[0] == "v":
                        logical = write[1]
                        current = mapped_vec_base(logical)
                        conflicts = any(
                            max(
                                reg_last_read[current + vi],
                                reg_last_write[current + vi] + 1,
                            )
                            > cycle
                            for vi in range(VLEN)
                        )
                        if conflicts:
                            renamed = alloc_words(VLEN)
                            if renamed is not None:
                                allocations.append((write, renamed))
                                continue
                        for vi in range(VLEN):
                            delayed_cycle = max(
                                delayed_cycle,
                                reg_last_read[current + vi],
                                reg_last_write[current + vi] + 1,
                            )
                    elif write[0] == "s":
                        logical = write[1]
                        current = phys[logical]
                        conflict_cycle = max(
                            reg_last_read[current],
                            reg_last_write[current] + 1,
                        )
                        if conflict_cycle > cycle and logical not in vector_addrs:
                            renamed = alloc_words(1)
                            if renamed is not None:
                                allocations.append((write, renamed))
                                continue
                        delayed_cycle = max(delayed_cycle, conflict_cycle)
                    elif write[0] == "lane":
                        current = phys[write[1] + write[2]]
                        delayed_cycle = max(
                            delayed_cycle,
                            reg_last_read[current],
                            reg_last_write[current] + 1,
                        )

                if allocations:
                    for write, renamed in allocations:
                        if write[0] == "v":
                            for vi in range(VLEN):
                                phys[write[1] + vi] = renamed + vi
                        elif write[0] == "s":
                            phys[write[1]] = renamed
                    break
                if delayed_cycle > cycle:
                    cycle = delayed_cycle
                    continue
                break

            rewritten = rewrite_slot(engine, slot)
            schedule[cycle].append((engine, rewritten))
            usage[cycle][engine] += 1

            reads, writes = self.slot_rw(engine, rewritten)
            for addr in reads:
                reg_last_read[addr] = max(reg_last_read[addr], cycle)
            for addr in writes:
                reg_avail[addr] = cycle + 1
                reg_last_write[addr] = cycle

        self.scratch_ptr = max(self.scratch_ptr, next_free)
        if not schedule:
            return []
        res = []
        for cycle in range(max(schedule) + 1):
            bundle = {}
            for engine, slot in schedule[cycle]:
                bundle.setdefault(engine, []).append(slot)
            res.append(bundle)
        return res

    def optimize_schedule(self):
        pause_idxs = [
            i
            for i, instr in enumerate(self.instrs)
            if any(
                engine == "flow" and any(slot[0] == "pause" for slot in slots)
                for engine, slots in instr.items()
            )
        ]
        if len(pause_idxs) < 2:
            return
        first_pause = pause_idxs[0]
        final_pause = pause_idxs[-1]
        segment = self.instrs[first_pause + 1 : final_pause]
        if self.use_virtual_tmp_allocator:
            if self.virtual_schedule_chunk_size > 0:
                scheduled = []
                for start in range(
                    0, len(segment), self.virtual_schedule_chunk_size
                ):
                    scheduled.extend(
                        self.schedule_segment(
                            segment[start : start + self.virtual_schedule_chunk_size]
                        )
                    )
                scheduled = self.color_virtual_tmp_vectors(scheduled)
            else:
                for _attempt in range(16):
                    self.virtual_color_failure = None
                    scheduled = (
                        self.schedule_segment_dag(segment)
                        if self.use_dag_scheduler
                        else self.schedule_segment(segment)
                    )
                    try:
                        scheduled = self.color_virtual_tmp_vectors(scheduled)
                        break
                    except AssertionError:
                        if self.virtual_color_failure is None:
                            raise
                        base, start, _needed = self.virtual_color_failure
                        self.virtual_start_delays[base] = max(
                            self.virtual_start_delays.get(base, 0), start + 8
                        )
                else:
                    raise AssertionError("Virtual tmp pressure repair did not converge")
        else:
            scheduled = (
                self.schedule_segment_dag(segment)
                if self.use_dag_scheduler
                else self.schedule_segment_renamed(segment)
            )
        self.instrs = (
            self.instrs[: first_pause + 1]
            + scheduled
            + self.instrs[final_pause:]
        )

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
        level3_split_value = (
            12
            if self.use_biased_heap_index
            else init_values["forest_values_p"] + 11
        )
        root_child_base_value = (
            init_values["forest_values_p"] + 2
            if self.use_bias_free_c5
            else init_values["forest_values_p"] + 1
        )
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
        max_groups = min(n_vec_groups, self.max_groups_limit)
        use_level3_cache = True
        if self.use_biased_heap_index:
            assert not self.use_level3_descriptor_select
            assert not self.use_level4_scalar_cache
        if self.reuse_addr_as_tmpa:
            assert not self.use_level4_scalar_cache
        level3_rounds = set()
        level3_round3_groups = set(self.level3_round3_groups)
        level3_round14_groups = set(self.level3_round14_groups)
        base_ready_done_threshold = self.base_ready_done_threshold
        base_ready_valu_limit = self.base_ready_valu_limit
        hash_half_alu_done_threshold = self.hash_half_alu_done_threshold
        zero_skip_done_threshold = 13
        idx = [self.alloc_scratch(f"idx{g}", VLEN) for g in range(max_groups)]
        val = [self.alloc_scratch(f"val{g}", VLEN) for g in range(max_groups)]
        addr = [self.alloc_scratch(f"addr{g}", VLEN) for g in range(max_groups)]
        tmpa = (
            addr
            if self.reuse_addr_as_tmpa
            else [
                self.alloc_scratch(f"tmpa{g}", VLEN)
                for g in range(max_groups)
            ]
        )
        tmpb_pool_size = self.tmpb_pool_size
        tmpb = [self.alloc_scratch(f"tmpb{i}", VLEN) for i in range(tmpb_pool_size)]
        virtual_tmp_spills = [
            self.alloc_scratch(f"virtual_tmp_spill{i}", VLEN)
            for i in range(self.virtual_tmp_spill_vectors)
        ]
        self.tmpb_physical_bases = list(tmpb) + virtual_tmp_spills
        if self.use_dedicated_l3_prep_pool:
            assert not self.use_level4_scalar_cache
            l3_prep_pool = [
                [
                    self.alloc_scratch(f"l3_prep_bank{bank}_{name}", VLEN)
                    for name in ("pair", "half", "selected", "work")
                ]
                for bank in range(self.l3_prep_bank_count)
            ]
        else:
            l3_prep_pool = None
        store_ptr = [self.alloc_scratch(f"store_ptr{g}") for g in range(max_groups)]
        preloaded_value_count = 0

        def fill_setup_preloads(loads):
            nonlocal preloaded_value_count
            loads = list(loads)
            while (
                setup_phase
                and preloaded_value_count < min(max_groups, 2)
                and len(loads) < SLOT_LIMITS["load"]
            ):
                loads.append(
                    (
                        "vload",
                        val[preloaded_value_count],
                        store_ptr[preloaded_value_count],
                    )
                )
                preloaded_value_count += 1
            return loads

        setup_flow_slots = [
            ("add_imm", store_ptr[g], self.scratch["inp_values_p"], g * VLEN)
            for g in range(max_groups)
        ]
        setup_phase = True
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        root_child_base_v = self.alloc_scratch("root_child_base_v", VLEN)
        addr_update_const_v = self.alloc_scratch("addr_update_const_v", VLEN)
        addr_update_odd_v = (
            None
            if self.use_additive_index_update
            else self.alloc_scratch("addr_update_odd_v", VLEN)
        )
        root_value_v = (
            None
            if self.use_scalar_root_value
            else self.alloc_scratch("root_value_v", VLEN)
        )
        root_value_initial_v = (
            self.alloc_scratch("root_value_initial_v", VLEN)
            if self.use_bias_free_c5 and self.bias_keep_initial_unbiased
            else None
        )
        root_value_s = (
            self.alloc_scratch("root_value_s")
            if self.use_scalar_root_value
            else None
        )
        level1_diff = self.alloc_scratch("level1_diff")
        level1_right_v = self.alloc_scratch("level1_right_v", VLEN)
        level1_diff_v = self.alloc_scratch("level1_diff_v", VLEN)
        level2_right_v = [
            self.alloc_scratch(f"level2_right_v{i}", VLEN) for i in range(2)
        ]
        level2_diff_v = [
            self.alloc_scratch(f"level2_diff_v{i}", VLEN) for i in range(2)
        ]
        level3_base_v = [
            self.alloc_scratch(f"level3_base_v{i}", VLEN) for i in range(4)
        ]
        level3_diff_v = [
            self.alloc_scratch(f"level3_diff_v{i}", VLEN) for i in range(4)
        ]
        level3_split_v = (
            None
            if self.use_level3_descriptor_select
            else self.alloc_scratch("level3_split_v1", VLEN)
        )
        if self.level3_prep_alu_hybrid:
            level3_base_pair_delta = [
                self.alloc_scratch(f"level3_base_pair_delta{i}", VLEN)
                for i in range(2)
            ]
            level3_diff_pair_delta = [
                self.alloc_scratch(f"level3_diff_pair_delta{i}", VLEN)
                for i in range(2)
            ]
        else:
            level3_base_pair_delta = None
            level3_diff_pair_delta = None
        if self.use_level4_scalar_cache:
            level4_nodes = self.alloc_scratch("level4_nodes", 16)
            level4_diff = self.alloc_scratch("level4_diff", 8)
        else:
            level4_nodes = None
            level4_diff = None
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
                    if c in hash_mult_v:
                        hash_const_v[c] = hash_mult_v[c]
                    else:
                        hash_const_v[c] = self.alloc_scratch(f"hash_const_{c:x}", VLEN)

        # Fuse hash stages 2 and 3.  With 32-bit wrapping arithmetic:
        #   y = 33*x + c2
        #   z = (y + c3) ^ (y << 9)
        #     = (33*x + c2 + c3) ^ ((33 << 9)*x + (c2 << 9))
        # This replaces four vector operations with three for every hash.
        hash23_mult_lo = 33
        hash23_mult_hi = 33 << 9
        hash23_const_lo = (HASH_STAGES[2][1] + HASH_STAGES[3][1]) % (2**32)
        hash23_const_hi = (HASH_STAGES[2][1] << 9) % (2**32)
        hash23_mult_lo_v = hash_mult_v[hash23_mult_lo]
        hash23_mult_hi_v = self.alloc_scratch("hash23_mult_hi", VLEN)
        hash23_const_lo_v = self.alloc_scratch("hash23_const_lo", VLEN)
        hash23_const_hi_v = self.alloc_scratch("hash23_const_hi", VLEN)
        scalar_const_values = [
            1,
            2,
            4,
            8,
            root_child_base_value,
            1 - init_values["forest_values_p"],
            2 - init_values["forest_values_p"],
            *hash_const_v.keys(),
            *hash_mult_v.keys(),
            init_values["forest_values_p"] + 8,
            level3_split_value,
            init_values["forest_values_p"] + 15,
            init_values["forest_values_p"] + 23,
            hash23_mult_hi,
            hash23_const_lo,
            hash23_const_hi,
        ]
        scalar_const_addrs = {}
        scalar_const_loads = []
        derived_level4_constants = (
            {
                4,
                8,
                init_values["forest_values_p"] + 15,
                init_values["forest_values_p"] + 23,
            }
            if self.use_level4_scalar_cache
            else set()
        )
        for c in scalar_const_values:
            if c in scalar_const_addrs:
                continue
            addr_c = self.reserve_const(c)
            scalar_const_addrs[c] = addr_c
            if c not in derived_level4_constants:
                scalar_const_loads.append(("const", addr_c, c))
        one_const = scalar_const_addrs[1]
        two_const = scalar_const_addrs[2]
        setup_nodes_low = tmpa[0]
        setup_nodes_high = tmpa[1]

        early_const_values = {
            1,
            2,
            init_values["forest_values_p"] + 1,
            root_child_base_value,
            init_values["forest_values_p"] + 8,
            1 - init_values["forest_values_p"],
            2 - init_values["forest_values_p"],
            level3_split_value,
        }
        remaining_const_loads = [
            ("const", scalar_const_addrs[c], c)
            for c in scalar_const_values
            if c not in early_const_values and c not in derived_level4_constants
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
                ("vload", setup_nodes_low, self.scratch["forest_values_p"]),
                ("const", one_const, 1),
            ],
            valu=(
                [("vbroadcast", root_value_initial_v, setup_nodes_low)]
                if root_value_initial_v is not None
                else []
            ),
        )
        if self.use_bias_free_c5:
            emit(
                load=[
                    (
                        "const",
                        scalar_const_addrs[0xB55A4F09],
                        0xB55A4F09,
                    )
                ]
            )
            emit(
                alu=[
                    (
                        "^",
                        setup_nodes_low + lane,
                        setup_nodes_low + lane,
                        scalar_const_addrs[0xB55A4F09],
                    )
                    for lane in range(VLEN)
                ]
            )
        emit(
            load=[
                ("const", two_const, 2),
                (
                    "const",
                    scalar_const_addrs[root_child_base_value],
                    root_child_base_value,
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
            load=fill_setup_preloads([
                (
                    "const",
                    scalar_const_addrs[init_values["forest_values_p"] + 8],
                    init_values["forest_values_p"] + 8,
                ),
            ]),
            valu=[
                ("vbroadcast", one_v, one_const),
                ("vbroadcast", two_v, two_const),
                (
                    "vbroadcast",
                    root_child_base_v,
                    scalar_const_addrs[root_child_base_value],
                ),
                (
                    "vbroadcast",
                    addr_update_const_v,
                    scalar_const_addrs[1 - init_values["forest_values_p"]],
                ),
                *(
                    [
                        (
                            "vbroadcast",
                            addr_update_odd_v,
                            scalar_const_addrs[2 - init_values["forest_values_p"]],
                        ),
                    ]
                    if addr_update_odd_v is not None
                    else []
                ),
                *(
                    [("vbroadcast", root_value_v, setup_nodes_low)]
                    if root_value_v is not None
                    else []
                ),
            ]
        )
        level1_base_lane = (
            setup_nodes_low + 2
            if self.use_bias_free_c5 and not self.reuse_addr_as_tmpa
            else setup_nodes_low + 1
        )
        level1_other_lane = (
            setup_nodes_low + 1
            if self.use_bias_free_c5 and not self.reuse_addr_as_tmpa
            else setup_nodes_low + 2
        )
        emit(
            load=take_const_loads(2),
            alu=[("-", level1_diff, level1_other_lane, level1_base_lane)],
            valu=[("vbroadcast", level1_right_v, level1_base_lane)],
        )
        emit(
            load=take_const_loads(2),
            valu=[
                ("vbroadcast", level1_diff_v, level1_diff),
                ("vbroadcast", hash_const_v[0x7ED55D16], scalar_const_addrs[0x7ED55D16]),
            ],
        )
        level2_pairs = [(3, 4), (5, 6)]
        for pair_i, (left_node, right_node) in enumerate(level2_pairs):
            if self.use_bias_free_c5 and not self.reuse_addr_as_tmpa:
                left_node, right_node = right_node, left_node
            left_lane = setup_nodes_low + left_node
            right_lane = setup_nodes_low + right_node
            emit(
                load=fill_setup_preloads([]),
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
                load=take_const_loads(2),
                alu=[("-", level1_diff, right_lane, left_lane)],
                valu=[("vbroadcast", level2_right_v[pair_i], left_lane)],
            )
            emit(
                load=take_const_loads(2),
                valu=[
                    ("vbroadcast", level2_diff_v[pair_i], level1_diff),
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
                        if (
                            pair_i == 1
                            and 9 in hash_mult_v
                            and hash_const_v.get(9) != hash_mult_v[9]
                        )
                        else []
                    ),
                ]
            )
        emit(
            load=fill_setup_preloads([
                (
                    "const",
                    scalar_const_addrs[level3_split_value],
                    level3_split_value,
                ),
            ]),
        )
        level3_pairs = [(7, 8), (9, 10), (11, 12), (13, 14)]
        deferred_setup_ops = []
        emit(
            load=fill_setup_preloads([
                (
                    "vload",
                    setup_nodes_high,
                    scalar_const_addrs[init_values["forest_values_p"] + 8],
                ),
            ]),
            valu=(
                [
                    (
                        "vbroadcast",
                        level3_split_v,
                        scalar_const_addrs[level3_split_value],
                    ),
                ]
                if level3_split_v is not None
                else []
            ),
        )
        if self.use_bias_free_c5:
            emit(
                alu=[
                    (
                        "^",
                        setup_nodes_high + lane,
                        setup_nodes_high + lane,
                        scalar_const_addrs[0xB55A4F09],
                    )
                    for lane in range(7)
                ]
            )
        for pair_i, (even_node, odd_node) in enumerate(level3_pairs):
            if pair_i == 0:
                even_src = setup_nodes_low + 7
                odd_src = setup_nodes_high
            else:
                even_src = setup_nodes_high + pair_i * 2 - 1
                odd_src = even_src + 1
            next_pair = (
                level3_pairs[pair_i + 1] if pair_i + 1 < len(level3_pairs) else None
            )
            base_broadcasts = [("vbroadcast", level3_base_v[pair_i], even_src)]
            diff_broadcast = ("vbroadcast", level3_diff_v[pair_i], level1_diff)
            if next_pair is None:
                deferred_setup_ops.append(
                    {
                        "alu": [("-", level1_diff, odd_src, even_src)],
                        "valu": base_broadcasts,
                    }
                )
                deferred_setup_ops.append({"valu": [diff_broadcast]})
            else:
                emit(
                    load=fill_setup_preloads(take_const_loads(2)),
                    alu=[("-", level1_diff, odd_src, even_src)],
                    valu=base_broadcasts,
                )
                emit(
                    load=fill_setup_preloads(take_const_loads(2)),
                    valu=[diff_broadcast]
                )
        while remaining_const_loads:
            emit(load=fill_setup_preloads(take_const_loads(SLOT_LIMITS["load"])))

        if self.use_level4_scalar_cache:
            emit(
                alu=[("+", scalar_const_addrs[4], two_const, two_const)],
                flow=[
                    (
                        "add_imm",
                        scalar_const_addrs[init_values["forest_values_p"] + 15],
                        self.scratch["forest_values_p"],
                        15,
                    )
                ],
            )
            emit(
                alu=[
                    (
                        "+",
                        scalar_const_addrs[8],
                        scalar_const_addrs[4],
                        scalar_const_addrs[4],
                    )
                ],
                flow=[
                    (
                        "add_imm",
                        scalar_const_addrs[init_values["forest_values_p"] + 23],
                        self.scratch["forest_values_p"],
                        23,
                    )
                ],
            )
            emit(
                load=[
                    (
                        "vload",
                        level4_nodes,
                        scalar_const_addrs[init_values["forest_values_p"] + 15],
                    ),
                    (
                        "vload",
                        level4_nodes + VLEN,
                        scalar_const_addrs[init_values["forest_values_p"] + 23],
                    ),
                ]
            )
            if self.use_bias_free_c5:
                for start in (0, VLEN):
                    emit(
                        alu=[
                            (
                                "^",
                                level4_nodes + start + lane,
                                level4_nodes + start + lane,
                                scalar_const_addrs[0xB55A4F09],
                            )
                            for lane in range(VLEN)
                        ]
                    )
            emit(
                alu=[
                    (
                        "-",
                        level4_diff + i,
                        level4_nodes + i * 2 + 1,
                        level4_nodes + i * 2,
                    )
                    for i in range(8)
                ]
            )
        if self.level3_prep_alu_hybrid or self.use_virtual_tmp_allocator:
            for setup_op in deferred_setup_ops:
                emit(**setup_op)
            if self.level3_prep_alu_hybrid:
                emit(
                    valu=[
                        (
                            "-",
                            level3_base_pair_delta[0],
                            level3_base_v[1],
                            level3_base_v[0],
                        ),
                        (
                            "-",
                            level3_base_pair_delta[1],
                            level3_base_v[3],
                            level3_base_v[2],
                        ),
                        (
                            "-",
                            level3_diff_pair_delta[0],
                            level3_diff_v[1],
                            level3_diff_v[0],
                        ),
                        (
                            "-",
                            level3_diff_pair_delta[1],
                            level3_diff_v[3],
                            level3_diff_v[2],
                        ),
                    ]
                )
            pending_setup_ops = []
        else:
            pending_setup_ops = deferred_setup_ops
        if self.use_scalar_root_value:
            emit(
                load=[
                    ("load", root_value_s, self.scratch["forest_values_p"]),
                ]
            )
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
        setup_broadcasts.extend(
            [
                ("vbroadcast", hash23_mult_hi_v, scalar_const_addrs[hash23_mult_hi]),
                ("vbroadcast", hash23_const_lo_v, scalar_const_addrs[hash23_const_lo]),
                ("vbroadcast", hash23_const_hi_v, scalar_const_addrs[hash23_const_hi]),
            ]
        )
        pending_setup_broadcasts = setup_broadcasts

        level4_program = []
        if self.use_level4_scalar_cache:
            level4_program.extend(
                [
                    (
                        "alu",
                        "-",
                        "addr",
                        "idx",
                        scalar_const_addrs[init_values["forest_values_p"] + 15],
                    ),
                    ("alu", "&", "idx", "addr", one_const),
                    ("alu", "&", "tmpa", "addr", two_const),
                    ("alu", "&", "t0", "addr", scalar_const_addrs[4]),
                    ("alu", "&", "t1", "addr", scalar_const_addrs[8]),
                ]
            )
            level4_conditions = ["idx", "tmpa", "t0", "t1"]

            def emit_level4_pair(pair_i, dest):
                level4_program.append(
                    (
                        "alu",
                        "*",
                        dest,
                        level4_conditions[0],
                        level4_diff + pair_i,
                    )
                )
                level4_program.append(
                    (
                        "alu",
                        "+",
                        dest,
                        dest,
                        level4_nodes + pair_i * 2,
                    )
                )

            def emit_level4_tree(start, count, dest, free):
                if count == 1:
                    emit_level4_pair(start, dest)
                    return
                half = count // 2
                temp = free[0]
                emit_level4_tree(start, half, dest, free)
                emit_level4_tree(start + half, half, temp, free[1:])
                condition = level4_conditions[count.bit_length() - 1]
                level4_program.append(
                    ("flow", dest, condition, temp, dest)
                )

            emit_level4_tree(0, 8, "addr", ["t2", "t3", "t4"])

        precomputed_store_ptr_count = max_groups - len(setup_flow_slots)
        setup_phase = False
        next_virtual_tmp_base = SCRATCH_SIZE

        for block_start in range(0, n_vec_groups, max_groups):
            group_count = min(max_groups, n_vec_groups - block_start)

            states = [
                {
                    "round": 0,
                    "phase": "waiting_load",
                    "ready": 0,
                    "off": 0,
                    "base_ready": False,
                    "store_ready": False,
                    "tmpb_slot": None,
                    "tmpb_slot2": None,
                    "tmpb_slot3": None,
                    "tmpb_slot4": None,
                    "tmpb_slot5": None,
                    "l4_pc": 0,
                    "l3_prep_state": None,
                    "l3_prep_ready": 0,
                    "l3_prep_bank": None,
                }
                for _ in range(group_count)
            ]
            def advance_after_update(g, ready_cycle):
                states[g]["round"] += 1
                states[g]["ready"] = ready_cycle
                states[g]["off"] = 0
                states[g]["base_ready"] = False
                if states[g]["round"] >= rounds:
                    states[g]["phase"] = "store" if states[g]["store_ready"] else "store_addr"
                elif use_level3_cache and (
                    states[g]["round"] in level3_rounds
                    or (
                        states[g]["round"] == 14
                        and (block_start + g) in level3_round14_groups
                    )
                    or (
                        states[g]["round"] == 3
                        and (block_start + g) in level3_round3_groups
                    )
                ):
                    if (
                        self.use_level3_descriptor_select
                        and states[g]["l3_prep_state"] == "ready"
                    ):
                        states[g]["phase"] = (
                            "level3_desc_parity_prepared"
                            if self.recompute_prepared_parity
                            else "level3_desc_delay_prepared"
                        )
                    elif (
                        self.use_level3_descriptor_select
                        and states[g]["l3_prep_state"] is not None
                    ):
                        states[g]["phase"] = "level3_desc_parity_wait"
                    else:
                        states[g]["phase"] = "addr"
                elif (
                    self.use_level4_scalar_cache
                    and states[g]["round"] == 15
                    and block_start + g in self.level4_scalar_groups
                ):
                    states[g]["phase"] = "addr"
                elif states[g]["round"] % (forest_height + 1) >= 3:
                    states[g]["phase"] = (
                        "gather_addr"
                        if self.use_biased_heap_index
                        else "gather"
                    )
                else:
                    states[g]["phase"] = "addr"

            sched_cycle = 0
            active_precomputed_store_ptr_count = (
                precomputed_store_ptr_count if block_start == 0 else 0
            )
            init_load_pair = (
                min(preloaded_value_count, group_count) if block_start == 0 else 0
            )
            next_init_ptr = active_precomputed_store_ptr_count
            ptr_ready = [
                0 if g < active_precomputed_store_ptr_count else 10**9
                for g in range(group_count)
            ]
            for g in range(min(active_precomputed_store_ptr_count, group_count)):
                states[g]["store_ready"] = True
            for g in range(init_load_pair):
                states[g]["store_ready"] = True
                states[g]["phase"] = (
                    "initial_bias"
                    if self.use_bias_free_c5
                    and not self.bias_keep_initial_unbiased
                    else "addr"
                )
                states[g]["ready"] = 0
            while any(st["phase"] != "done" for st in states):
                done_count = sum(st["phase"] == "done" for st in states)
                tmpb_occupied_at_cycle_start = {
                    st[key]
                    for st in states
                    for key in (
                        "tmpb_slot",
                        "tmpb_slot2",
                        "tmpb_slot3",
                        "tmpb_slot4",
                        "tmpb_slot5",
                    )
                    if st[key] is not None
                }
                if done_count >= 19:
                    tail_load_priority_global = self.tail_load_priority
                    tail_load_priority = [
                        gg - block_start
                        for gg in tail_load_priority_global
                        if block_start <= gg < block_start + group_count
                    ]
                    tail_load_set = set(tail_load_priority)
                    load_scan_order = tail_load_priority + [
                        g
                        for g in range(group_count - 1, -1, -1)
                        if g not in tail_load_set
                    ]
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
                if done_count >= 17:
                    valu_scan_order = list(range(group_count - 1, -1, -1))
                else:
                    valu_scan_order = list(range(group_count))
                load_slots = []
                alu_slots = []
                valu_slots = []
                store_slots = []
                flow_slots = []

                def add_alu_vec(op, dest, a1, a2):
                    if op not in ("+", "-", "*", "^", "&", "<<", ">>", "<"):
                        return False
                    if len(alu_slots) + VLEN > SLOT_LIMITS["alu"]:
                        return False
                    alu_slots.extend(
                        (op, dest + vi, a1 + vi, a2 + vi) for vi in range(VLEN)
                    )
                    return True

                def add_alu_vec_lanes(op, dest, a1, a2, start, count):
                    if op not in ("+", "-", "^", "&", "<<", ">>", "<"):
                        return False
                    if len(alu_slots) + count > SLOT_LIMITS["alu"]:
                        return False
                    alu_slots.extend(
                        (op, dest + vi, a1 + vi, a2 + vi)
                        for vi in range(start, start + count)
                    )
                    return True

                def add_alu_vec_scalar(op, dest, a1, scalar):
                    if len(alu_slots) + VLEN > SLOT_LIMITS["alu"]:
                        return False
                    alu_slots.extend(
                        (op, dest + vi, a1 + vi, scalar) for vi in range(VLEN)
                    )
                    return True

                def add_simple_vec(op, dest, a1, a2):
                    if (
                        done_count >= self.simple_valu_first_done_threshold
                        and len(valu_slots) <= self.simple_valu_first_limit
                    ):
                        valu_slots.append((op, dest, a1, a2))
                        return True
                    # Prefer ALU for simple ops so VALU stays free for multiply_add.
                    # ALU has 12 slots but VLEN=8, so at most one spilled vector/cycle
                    # (alu-8..11 stay idle — can't fit another full vector).
                    if add_alu_vec(op, dest, a1, a2):
                        return True
                    if len(valu_slots) < SLOT_LIMITS["valu"]:
                        valu_slots.append((op, dest, a1, a2))
                        return True
                    return False

                def tmpb_slots_written_this_cycle():
                    written = set()
                    for engine, slots in (
                        ("load", load_slots),
                        ("alu", alu_slots),
                        ("valu", valu_slots),
                        ("flow", flow_slots),
                    ):
                        for slot in slots:
                            _reads, writes = self.slot_rw(engine, slot)
                            for slot_i, base in enumerate(tmpb):
                                if any(base <= addr < base + VLEN for addr in writes):
                                    written.add(slot_i)
                    return written

                def alloc_virtual_tmpb_slots(count):
                    nonlocal next_virtual_tmp_base
                    def token_for(handle):
                        return self.virtual_tmp_tokens.get(handle, handle)

                    used_tokens = {
                        token_for(handle)
                        for handle in tmpb_slots_written_this_cycle()
                    }
                    if (
                        self.preschedule_level3_descriptor
                        or self.reuse_addr_as_tmpa
                    ):
                        used_tokens.update(
                            token_for(handle)
                            for handle in tmpb_occupied_at_cycle_start
                        )
                    used_tokens.update(
                        token_for(state[key])
                        for state in states
                        for key in (
                            "tmpb_slot",
                            "tmpb_slot2",
                            "tmpb_slot3",
                            "tmpb_slot4",
                            "tmpb_slot5",
                        )
                        if state[key] is not None
                    )
                    free_tokens = [
                        token
                        for token in range(self.virtual_tmp_capacity)
                        if token not in used_tokens
                    ]
                    if len(free_tokens) < count:
                        return None
                    result = []
                    for token in free_tokens[:count]:
                        base = next_virtual_tmp_base
                        next_virtual_tmp_base += VLEN
                        self.virtual_vector_bases.add(base)
                        tmpb.append(base)
                        handle = len(tmpb) - 1
                        self.virtual_tmp_tokens[handle] = token
                        result.append(handle)
                    return result

                def alloc_tmpb_slot():
                    if self.use_virtual_tmp_allocator:
                        slots = alloc_virtual_tmpb_slots(1)
                        return slots[0] if slots is not None else None
                    used = tmpb_slots_written_this_cycle()
                    if (
                        self.preschedule_level3_descriptor
                        or self.reuse_addr_as_tmpa
                    ):
                        used.update(tmpb_occupied_at_cycle_start)
                    for state in states:
                        for key in (
                            "tmpb_slot",
                            "tmpb_slot2",
                            "tmpb_slot3",
                            "tmpb_slot4",
                            "tmpb_slot5",
                        ):
                            if state[key] is not None:
                                used.add(state[key])
                    for slot_i in range(tmpb_pool_size):
                        if slot_i not in used:
                            return slot_i
                    return None

                def alloc_two_tmpb_slots():
                    slots = alloc_tmpb_slots(2)
                    return tuple(slots) if slots is not None else None

                def alloc_tmpb_slots(count):
                    if self.use_virtual_tmp_allocator:
                        return alloc_virtual_tmpb_slots(count)
                    used = tmpb_slots_written_this_cycle()
                    if (
                        self.preschedule_level3_descriptor
                        or self.reuse_addr_as_tmpa
                    ):
                        used.update(tmpb_occupied_at_cycle_start)
                    for state in states:
                        for key in (
                            "tmpb_slot",
                            "tmpb_slot2",
                            "tmpb_slot3",
                            "tmpb_slot4",
                            "tmpb_slot5",
                        ):
                            if state[key] is not None:
                                used.add(state[key])
                    result = []
                    for slot_i in range(tmpb_pool_size):
                        if slot_i not in used:
                            result.append(slot_i)
                            if len(result) == count:
                                return result
                    return None

                def level4_reg(g, name):
                    if name == "idx":
                        return idx[g]
                    if name == "addr":
                        return addr[g]
                    if name == "tmpa":
                        return tmpa[g]
                    slot_keys = {
                        "t0": "tmpb_slot",
                        "t1": "tmpb_slot2",
                        "t2": "tmpb_slot3",
                        "t3": "tmpb_slot4",
                        "t4": "tmpb_slot5",
                    }
                    slot = states[g][slot_keys[name]]
                    if slot is None:
                        raise AssertionError("Missing L4 temporary")
                    return tmpb[slot]

                def l3_prep_regs(st):
                    if self.use_dedicated_l3_prep_pool:
                        bank = st["l3_prep_bank"]
                        if bank is None:
                            return None
                        return tuple(l3_prep_pool[bank])
                    slots = (
                        st["tmpb_slot2"],
                        st["tmpb_slot3"],
                        st["tmpb_slot4"],
                        st["tmpb_slot5"],
                    )
                    if None in slots:
                        return None
                    return tuple(tmpb[slot] for slot in slots)

                def maybe_add_base_ready(g, st, r):
                    if (
                        not self.use_biased_heap_index
                        and not self.reuse_addr_as_tmpa
                        and
                        not st["base_ready"]
                        and r + 1 < rounds
                        and (r + 1) % (forest_height + 1) != 0
                        and r % (forest_height + 1) != 0
                        and done_count >= base_ready_done_threshold
                        and len(valu_slots) <= base_ready_valu_limit
                    ):
                        valu_slots.append(
                            (
                                "multiply_add",
                                addr[g],
                                idx[g],
                                two_v,
                                (
                                    addr_update_odd_v
                                    if self.use_bias_free_c5
                                    else addr_update_const_v
                                ),
                            )
                        )
                        st["base_ready"] = True

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
                            states[g]["phase"] = (
                                "initial_bias"
                                if self.use_bias_free_c5
                                and not self.bias_keep_initial_unbiased
                                else "addr"
                            )
                            states[g]["ready"] = sched_cycle + 1
                        init_load_pair += len(load_slots)
                else:
                    for g in load_scan_order:
                        st = states[g]
                        if len(load_slots) >= SLOT_LIMITS["load"]:
                            break
                        while (
                            done_count >= self.double_gather_done_threshold
                            and len(load_slots) < SLOT_LIMITS["load"]
                            and st["phase"] == "gather"
                            and st["ready"] <= sched_cycle
                            and st["off"] < VLEN
                        ):
                            load_slots.append(
                                (
                                    "load_offset",
                                    addr[g],
                                    addr[g] if self.use_biased_heap_index else idx[g],
                                    st["off"],
                                )
                            )
                            st["off"] += 1
                            if st["off"] == VLEN:
                                st["phase"] = (
                                    "bias_gathered_node"
                                    if self.use_bias_free_c5
                                    else "xor"
                                )
                                st["ready"] = sched_cycle + 1
                        if len(load_slots) >= SLOT_LIMITS["load"]:
                            break
                        if (
                            st["phase"] == "gather"
                            and st["ready"] <= sched_cycle
                            and st["off"] < VLEN
                        ):
                            load_slots.append(
                                (
                                    "load_offset",
                                    addr[g],
                                    addr[g] if self.use_biased_heap_index else idx[g],
                                    st["off"],
                                )
                            )
                            st["off"] += 1
                            if st["off"] == VLEN:
                                st["phase"] = (
                                    "bias_gathered_node"
                                    if self.use_bias_free_c5
                                    else "xor"
                                )
                                st["ready"] = sched_cycle + 1
                    if done_count < self.double_gather_done_threshold:
                        for g in load_scan_order:
                            st = states[g]
                            if len(load_slots) >= SLOT_LIMITS["load"]:
                                break
                            if (
                                st["phase"] == "gather"
                                and st["ready"] <= sched_cycle
                                and st["off"] < VLEN
                            ):
                                load_slots.append(
                                    (
                                        "load_offset",
                                        addr[g],
                                        addr[g]
                                        if self.use_biased_heap_index
                                        else idx[g],
                                        st["off"],
                                    )
                                )
                                st["off"] += 1
                                if st["off"] == VLEN:
                                    st["phase"] = (
                                        "bias_gathered_node"
                                        if self.use_bias_free_c5
                                        else "xor"
                                    )
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

                if pending_setup_ops:
                    setup_op = pending_setup_ops[0]
                    setup_alu = setup_op.get("alu", [])
                    setup_valu = setup_op.get("valu", [])
                    if (
                        len(alu_slots) + len(setup_alu) <= SLOT_LIMITS["alu"]
                        and len(valu_slots) + len(setup_valu) <= SLOT_LIMITS["valu"]
                    ):
                        alu_slots.extend(setup_alu)
                        valu_slots.extend(setup_valu)
                        pending_setup_ops.pop(0)

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

                critical_flow_phases = {
                    "level2_select_flow",
                    "level3_select01_flow",
                    "level3_select23_flow",
                    "level3_final_select_flow",
                    "level4_scalar",
                    "select_inc",
                }
                has_critical_flow = any(
                    st["ready"] <= sched_cycle
                    and (
                        st["phase"] in critical_flow_phases
                        or st["phase"].startswith("level3_desc_flow")
                    )
                    for st in states
                )
                for g in flow_scan_order:
                    st = states[g]
                    if flow_slots:
                        break
                    if st["ready"] > sched_cycle:
                        continue
                    prep_state = st["l3_prep_state"]
                    if (
                        isinstance(prep_state, int)
                        and st["l3_prep_ready"] <= sched_cycle
                        and (
                            not self.level3_prep_alu_hybrid
                            or prep_state in (4, 9)
                        )
                        and (
                            self.prep_flow_yield_stride <= 0
                            or not has_critical_flow
                            or sched_cycle % self.prep_flow_yield_stride != 0
                        )
                    ):
                        prep_regs = l3_prep_regs(st)
                        if prep_regs is None:
                            continue
                        pair, half, selected, work = prep_regs
                        if self.level3_prep_alu_hybrid:
                            if prep_state == 4:
                                dest, cond, when_true, when_false = (
                                    selected,
                                    half,
                                    selected,
                                    work,
                                )
                            else:
                                dest, cond, when_true, when_false = (
                                    work,
                                    half,
                                    work,
                                    pair,
                                )
                        else:
                            prep_ops = [
                                (selected, pair, level3_base_v[1], level3_base_v[0]),
                                (work, pair, level3_base_v[3], level3_base_v[2]),
                                (selected, half, selected, work),
                                (work, pair, level3_diff_v[1], level3_diff_v[0]),
                                (pair, pair, level3_diff_v[3], level3_diff_v[2]),
                                (work, half, work, pair),
                            ]
                            dest, cond, when_true, when_false = prep_ops[prep_state]
                        flow_slots.append(
                            ("vselect", dest, cond, when_true, when_false)
                        )
                        prep_done = (
                            prep_state == 9
                            if self.level3_prep_alu_hybrid
                            else prep_state + 1 == len(prep_ops)
                        )
                        if prep_done:
                            st["l3_prep_state"] = "ready"
                            if st["phase"] == "wait_l3_prep":
                                st["phase"] = "level3_desc_madd_prepared"
                                st["ready"] = sched_cycle + 1
                        else:
                            st["l3_prep_state"] = prep_state + 1
                            st["l3_prep_ready"] = sched_cycle + 1
                    elif st["phase"] == "level2_select_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        if tmpb_slot is None:
                            continue
                        level2_cond = (
                            tmpb[tmpb_slot2]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                level2_cond,
                                (
                                    tmpb[tmpb_slot]
                                    if self.use_biased_heap_index
                                    else addr[g]
                                ),
                                (
                                    addr[g]
                                    if self.use_biased_heap_index
                                    else tmpb[tmpb_slot]
                                ),
                            )
                        )
                        st["tmpb_slot"] = None
                        if self.reuse_addr_as_tmpa:
                            st["tmpb_slot2"] = None
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_select01_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot3 = st["tmpb_slot3"]
                        if tmpb_slot is None:
                            continue
                        level3_cond = (
                            tmpb[tmpb_slot3]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                level3_cond,
                                (
                                    tmpb[tmpb_slot]
                                    if self.use_biased_heap_index
                                    else addr[g]
                                ),
                                (
                                    addr[g]
                                    if self.use_biased_heap_index
                                    else tmpb[tmpb_slot]
                                ),
                            )
                        )
                        st["phase"] = "level3_parity2"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_select23_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        tmpb_slot3 = st["tmpb_slot3"]
                        if tmpb_slot is None or tmpb_slot2 is None:
                            continue
                        level3_cond = (
                            tmpb[tmpb_slot3]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        flow_slots.append(
                            (
                                "vselect",
                                tmpb[tmpb_slot],
                                level3_cond,
                                (
                                    tmpb[tmpb_slot2]
                                    if self.use_biased_heap_index
                                    else tmpb[tmpb_slot]
                                ),
                                (
                                    tmpb[tmpb_slot]
                                    if self.use_biased_heap_index
                                    else tmpb[tmpb_slot2]
                                ),
                            )
                        )
                        st["phase"] = "level3_final_cmp"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level3_final_select_flow":
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        tmpb_slot3 = st["tmpb_slot3"]
                        if tmpb_slot is None:
                            continue
                        level3_cond = (
                            tmpb[tmpb_slot3]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                level3_cond,
                                addr[g],
                                tmpb[tmpb_slot],
                            )
                        )
                        st["tmpb_slot"] = None
                        st["tmpb_slot2"] = None
                        if self.reuse_addr_as_tmpa:
                            st["tmpb_slot3"] = None
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"].startswith("level3_desc_flow"):
                        tmpb_slot = st["tmpb_slot"]
                        tmpb_slot2 = st["tmpb_slot2"]
                        tmpb_slot3 = st["tmpb_slot3"]
                        if (
                            tmpb_slot is None
                            or tmpb_slot2 is None
                            or tmpb_slot3 is None
                        ):
                            continue
                        half = tmpb[tmpb_slot]
                        selected = tmpb[tmpb_slot2]
                        work = tmpb[tmpb_slot3]
                        step = int(st["phase"].removeprefix("level3_desc_flow"))
                        desc_ops = [
                            (selected, addr[g], level3_base_v[0], level3_base_v[1]),
                            (work, addr[g], level3_base_v[2], level3_base_v[3]),
                            (selected, half, selected, work),
                            (work, addr[g], level3_diff_v[0], level3_diff_v[1]),
                            (addr[g], addr[g], level3_diff_v[2], level3_diff_v[3]),
                            (work, half, work, addr[g]),
                        ]
                        dest, cond, when_true, when_false = desc_ops[step]
                        flow_slots.append(
                            ("vselect", dest, cond, when_true, when_false)
                        )
                        if step + 1 == len(desc_ops):
                            st["phase"] = "level3_desc_madd"
                        else:
                            st["phase"] = f"level3_desc_flow{step + 1}"
                        st["ready"] = sched_cycle + 1
                    elif st["phase"] == "level4_scalar":
                        op = level4_program[st["l4_pc"]]
                        if op[0] != "flow":
                            continue
                        _, dest, cond, when_true, when_false = op
                        flow_slots.append(
                            (
                                "vselect",
                                level4_reg(g, dest),
                                level4_reg(g, cond),
                                level4_reg(g, when_true),
                                level4_reg(g, when_false),
                            )
                        )
                        st["l4_pc"] += 1
                        st["ready"] = sched_cycle + 1
                        if st["l4_pc"] == len(level4_program):
                            st["tmpb_slot"] = None
                            st["tmpb_slot2"] = None
                            st["tmpb_slot3"] = None
                            st["tmpb_slot4"] = None
                            st["tmpb_slot5"] = None
                            st["phase"] = "xor"
                    elif st["phase"] == "select_inc":
                        assert addr_update_odd_v is not None
                        flow_slots.append(
                            (
                                "vselect",
                                addr[g],
                                tmpa[g],
                                (
                                    addr_update_const_v
                                    if self.use_bias_free_c5
                                    else addr_update_odd_v
                                ),
                                (
                                    addr_update_odd_v
                                    if self.use_bias_free_c5
                                    else addr_update_const_v
                                ),
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

                    if (
                        self.use_level3_descriptor_select
                        and self.preschedule_level3_descriptor
                        and r in (2, 13)
                        and r + 1 < rounds
                        and (
                            (
                                r == 2
                                and (block_start + g) in level3_round3_groups
                            )
                            or (
                                r == 13
                                and self.preschedule_level3_round14
                                and (block_start + g) in level3_round14_groups
                            )
                        )
                        and (
                            self.prep_start_hash_stage <= 0
                            or (
                                st["phase"].startswith("hash")
                                and st.get("hash_i", -1)
                                >= self.prep_start_hash_stage
                            )
                        )
                        and st["l3_prep_state"] is None
                    ):
                        if self.use_dedicated_l3_prep_pool:
                            used_banks = {
                                other["l3_prep_bank"]
                                for other in states
                                if other["l3_prep_bank"] is not None
                            }
                            free_bank = next(
                                (
                                    bank
                                    for bank in range(self.l3_prep_bank_count)
                                    if bank not in used_banks
                                ),
                                None,
                            )
                            if free_bank is not None:
                                st["l3_prep_bank"] = free_bank
                                st["l3_prep_state"] = "conditions"
                        else:
                            prep_slots = alloc_tmpb_slots(4)
                            if prep_slots is None:
                                prep_slots = ()
                        if not self.use_dedicated_l3_prep_pool and prep_slots:
                            (
                                st["tmpb_slot2"],
                                st["tmpb_slot3"],
                                st["tmpb_slot4"],
                                st["tmpb_slot5"],
                            ) = prep_slots
                            st["l3_prep_state"] = "conditions"
                    if (
                        st["l3_prep_state"] == "conditions"
                        and len(valu_slots) + 2 <= SLOT_LIMITS["valu"]
                    ):
                        prep_regs = l3_prep_regs(st)
                        if prep_regs is not None:
                            pair, half, _selected, _work = prep_regs
                            valu_slots.extend(
                                [
                                    ("&", pair, idx[g], one_v),
                                    ("&", half, idx[g], two_v),
                                ]
                            )
                            st["l3_prep_state"] = 0
                            st["l3_prep_ready"] = sched_cycle + 1

                    if phase == "level4_scalar":
                        op = level4_program[st["l4_pc"]]
                        if op[0] == "flow":
                            continue
                        _, alu_op, dest, src, scalar = op
                        if not add_alu_vec_scalar(
                            alu_op,
                            level4_reg(g, dest),
                            level4_reg(g, src),
                            scalar,
                        ):
                            continue
                        st["l4_pc"] += 1
                        st["ready"] = sched_cycle + 1
                        continue
                    if phase == "initial_bias":
                        bias_added = (
                            add_alu_vec(
                                "^",
                                val[g],
                                val[g],
                                hash_const_v[0xB55A4F09],
                            )
                            if self.bias_c5_edge_alu
                            else add_simple_vec(
                                "^",
                                val[g],
                                val[g],
                                hash_const_v[0xB55A4F09],
                            )
                        )
                        if not bias_added:
                            break
                        st["phase"] = "addr"
                        st["ready"] = sched_cycle + 1
                        continue
                    if phase == "bias_gathered_node":
                        bias_added = (
                            add_alu_vec(
                                "^",
                                addr[g],
                                addr[g],
                                hash_const_v[0xB55A4F09],
                            )
                            if self.bias_c5_gather_alu
                            else add_simple_vec(
                                "^",
                                addr[g],
                                addr[g],
                                hash_const_v[0xB55A4F09],
                            )
                        )
                        if not bias_added:
                            break
                        st["phase"] = "xor"
                        st["ready"] = sched_cycle + 1
                        continue
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
                        if self.use_bias_free_c5 and hi == 5:
                            hash5_shift_added = (
                                add_alu_vec(
                                    ">>",
                                    hash_tmp,
                                    val[g],
                                    hash_const_v[val3],
                                )
                                if self.bias_hash5_shift_alu
                                else add_simple_vec(
                                    ">>",
                                    hash_tmp,
                                    val[g],
                                    hash_const_v[val3],
                                )
                            )
                            if not hash5_shift_added:
                                continue
                            st["phase"] = "hash5_combine"
                            st["ready"] = sched_cycle + 1
                            continue
                        if hi == 2:
                            if len(valu_slots) + 2 > SLOT_LIMITS["valu"]:
                                continue
                            valu_slots.append(
                                (
                                    "multiply_add",
                                    tmpa[g],
                                    val[g],
                                    hash23_mult_lo_v,
                                    hash23_const_lo_v,
                                )
                            )
                            valu_slots.append(
                                (
                                    "multiply_add",
                                    hash_tmp,
                                    val[g],
                                    hash23_mult_hi_v,
                                    hash23_const_hi_v,
                                )
                            )
                            st["phase"] = "hash_combine"
                            maybe_add_base_ready(g, st, r)
                            st["ready"] = sched_cycle + 1
                            continue
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
                                if done_count >= zero_skip_done_threshold:
                                    advance_after_update(g, sched_cycle + 1)
                                else:
                                    st["phase"] = "zero"
                            else:
                                st["phase"] = "parity"
                            maybe_add_base_ready(g, st, r)
                            st["ready"] = sched_cycle + 1
                            continue
                        if len(valu_slots) + 2 > SLOT_LIMITS["valu"]:
                            if len(valu_slots) < SLOT_LIMITS["valu"]:
                                valu_slots.append((op1, tmpa[g], val[g], hash_const_v[val1]))
                                if add_alu_vec(op3, hash_tmp, val[g], hash_const_v[val3]):
                                    st["phase"] = "hash_combine"
                                elif (
                                    done_count >= hash_half_alu_done_threshold
                                    and add_alu_vec_lanes(
                                        op3, hash_tmp, val[g], hash_const_v[val3], 4, 4
                                    )
                                ):
                                    st["phase"] = "hash_pre_second_low4"
                                else:
                                    st["phase"] = "hash_pre_second"
                                maybe_add_base_ready(g, st, r)
                                st["ready"] = sched_cycle + 1
                            continue
                        valu_slots.append((op1, tmpa[g], val[g], hash_const_v[val1]))
                        valu_slots.append((op3, hash_tmp, val[g], hash_const_v[val3]))
                        st["phase"] = "hash_combine"
                        maybe_add_base_ready(g, st, r)
                        st["ready"] = sched_cycle + 1
                        continue
                    if phase == "addr":
                        if r % (forest_height + 1) == 0:
                            root_added = (
                                add_alu_vec_scalar(
                                    "^", val[g], val[g], root_value_s
                                )
                                if self.use_scalar_root_value
                                else add_simple_vec(
                                    "^",
                                    val[g],
                                    val[g],
                                    (
                                        root_value_initial_v
                                        if self.use_bias_free_c5
                                        and self.bias_keep_initial_unbiased
                                        and r == 0
                                        else root_value_v
                                    ),
                                )
                            )
                            if not root_added:
                                break
                            st["phase"] = "hash_pre"
                            st["hash_i"] = 0
                        elif r % (forest_height + 1) == 1:
                            if self.reuse_addr_as_tmpa:
                                if not add_simple_vec(
                                    "&", addr[g], idx[g], one_v
                                ):
                                    break
                                st["phase"] = "level1_select"
                                st["ready"] = sched_cycle + 1
                                continue
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
                            if self.reuse_addr_as_tmpa:
                                slots = alloc_tmpb_slots(2)
                                if slots is None:
                                    continue
                                tmpb_slot, tmpb_slot2 = slots
                                if not add_simple_vec(
                                    "&", tmpb[tmpb_slot2], idx[g], one_v
                                ):
                                    break
                                st["tmpb_slot"] = tmpb_slot
                                st["tmpb_slot2"] = tmpb_slot2
                                st["phase"] = "level2_pair0"
                                st["ready"] = sched_cycle + 1
                                continue
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
                        elif (
                            self.use_level4_scalar_cache
                            and r == 15
                            and block_start + g in self.level4_scalar_groups
                        ):
                            slots = alloc_tmpb_slots(5)
                            if slots is None:
                                continue
                            (
                                st["tmpb_slot"],
                                st["tmpb_slot2"],
                                st["tmpb_slot3"],
                                st["tmpb_slot4"],
                                st["tmpb_slot5"],
                            ) = slots
                            st["l4_pc"] = 0
                            st["phase"] = "level4_scalar"
                        elif use_level3_cache and (
                            r in level3_rounds
                            or (r == 14 and (block_start + g) in level3_round14_groups)
                            or (r == 3 and (block_start + g) in level3_round3_groups)
                        ):
                            if self.use_level3_descriptor_select:
                                slots = alloc_tmpb_slots(
                                    4 if self.reuse_addr_as_tmpa else 3
                                )
                                if slots is None:
                                    continue
                                (
                                    st["tmpb_slot"],
                                    st["tmpb_slot2"],
                                    st["tmpb_slot3"],
                                ) = slots[:3]
                                if self.reuse_addr_as_tmpa:
                                    st["tmpb_slot4"] = slots[3]
                                st["phase"] = "level3_desc_conditions"
                            else:
                                slots = alloc_tmpb_slots(
                                    3 if self.reuse_addr_as_tmpa else 2
                                )
                                if slots is None:
                                    continue
                                parity_dest = (
                                    tmpb[slots[2]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                )
                                if not add_simple_vec(
                                    "&", parity_dest, idx[g], one_v
                                ):
                                    break
                                st["tmpb_slot"] = slots[0]
                                st["tmpb_slot2"] = slots[1]
                                if self.reuse_addr_as_tmpa:
                                    st["tmpb_slot3"] = slots[2]
                                st["phase"] = "level3_pair0"
                        else:
                            st["phase"] = "gather"
                            st["ready"] = sched_cycle
                            continue
                        st["ready"] = sched_cycle + 1
                    elif phase == "gather_addr":
                        addr_added = (
                            add_alu_vec(
                                "-", addr[g], idx[g], addr_update_const_v
                            )
                            if self.biased_gather_addr_alu
                            else add_simple_vec(
                                "-", addr[g], idx[g], addr_update_const_v
                            )
                        )
                        if not addr_added:
                            break
                        st["phase"] = "gather"
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
                        parity_src = (
                            tmpb[st["tmpb_slot2"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                addr[g],
                                parity_src,
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
                                (
                                    tmpb[st["tmpb_slot2"]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                ),
                                level2_diff_v[1],
                                level2_right_v[1],
                            )
                        )
                        st["phase"] = "level2_cmp"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level2_cmp":
                        cmp_dest = (
                            tmpb[st["tmpb_slot2"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if not add_simple_vec("&", cmp_dest, idx[g], two_v):
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
                        parity_src = (
                            tmpb[st["tmpb_slot3"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", addr[g], parity_src, level3_diff_v[0], level3_base_v[0])
                        )
                        st["phase"] = "level3_pair1"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_conditions":
                        tmpb_slot = st["tmpb_slot"]
                        parity_dest = (
                            tmpb[st["tmpb_slot4"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if (
                            tmpb_slot is None
                            or len(valu_slots) + 2 > SLOT_LIMITS["valu"]
                            or len(alu_slots) + VLEN > SLOT_LIMITS["alu"]
                        ):
                            continue
                        valu_slots.extend(
                            [
                                ("&", parity_dest, idx[g], one_v),
                                ("&", addr[g], idx[g], two_v),
                            ]
                        )
                        alu_slots.extend(
                            (
                                "<",
                                tmpb[tmpb_slot] + lane,
                                idx[g] + lane,
                                scalar_const_addrs[level3_split_value],
                            )
                            for lane in range(VLEN)
                        )
                        st["phase"] = "level3_desc_flow0"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_parity_wait":
                        if not add_simple_vec("&", tmpa[g], idx[g], one_v):
                            break
                        st["phase"] = (
                            "level3_desc_madd_prepared"
                            if st["l3_prep_state"] == "ready"
                            else "wait_l3_prep"
                        )
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_parity_prepared":
                        if not add_simple_vec("&", tmpa[g], idx[g], one_v):
                            break
                        st["phase"] = "level3_desc_madd_prepared"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_delay_prepared":
                        st["phase"] = "level3_desc_madd_prepared"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_madd_prepared":
                        prep_regs = l3_prep_regs(st)
                        if prep_regs is None:
                            continue
                        _pair, _half, selected, work = prep_regs
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                addr[g],
                                tmpa[g],
                                work,
                                selected,
                            )
                        )
                        st["phase"] = "level3_desc_xor"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_madd":
                        tmpb_slot2 = st["tmpb_slot2"]
                        tmpb_slot3 = st["tmpb_slot3"]
                        if tmpb_slot2 is None or tmpb_slot3 is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                addr[g],
                                (
                                    tmpb[st["tmpb_slot4"]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                ),
                                tmpb[tmpb_slot3],
                                tmpb[tmpb_slot2],
                            )
                        )
                        st["phase"] = "level3_desc_xor_normal"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_xor_normal":
                        if not add_simple_vec("^", val[g], val[g], addr[g]):
                            break
                        st["tmpb_slot"] = None
                        st["tmpb_slot2"] = None
                        st["tmpb_slot3"] = None
                        if self.reuse_addr_as_tmpa:
                            st["tmpb_slot4"] = None
                        st["phase"] = "hash_pre"
                        st["hash_i"] = 0
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_desc_xor":
                        if not add_simple_vec("^", val[g], val[g], addr[g]):
                            break
                        st["tmpb_slot2"] = None
                        st["tmpb_slot3"] = None
                        st["tmpb_slot4"] = None
                        st["tmpb_slot5"] = None
                        st["l3_prep_state"] = None
                        st["l3_prep_bank"] = None
                        st["phase"] = "hash_pre"
                        st["hash_i"] = 0
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_pair1":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            (
                                "multiply_add",
                                tmpb[tmpb_slot],
                                (
                                    tmpb[st["tmpb_slot3"]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                ),
                                level3_diff_v[1],
                                level3_base_v[1],
                            )
                        )
                        st["phase"] = "level3_cmp01"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_cmp01":
                        cmp_dest = (
                            tmpb[st["tmpb_slot3"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if not add_simple_vec("&", cmp_dest, idx[g], two_v):
                            break
                        st["phase"] = "level3_select01_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_parity2":
                        parity_dest = (
                            tmpb[st["tmpb_slot3"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if not add_simple_vec("&", parity_dest, idx[g], one_v):
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
                            (
                                "multiply_add",
                                tmpb[tmpb_slot],
                                (
                                    tmpb[st["tmpb_slot3"]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                ),
                                level3_diff_v[2],
                                level3_base_v[2],
                            )
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
                            (
                                "multiply_add",
                                tmpb[tmpb_slot2],
                                (
                                    tmpb[st["tmpb_slot3"]]
                                    if self.reuse_addr_as_tmpa
                                    else tmpa[g]
                                ),
                                level3_diff_v[3],
                                level3_base_v[3],
                            )
                        )
                        st["phase"] = "level3_cmp23"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_cmp23":
                        cmp_dest = (
                            tmpb[st["tmpb_slot3"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if not add_simple_vec("&", cmp_dest, idx[g], two_v):
                            break
                        st["phase"] = "level3_select23_flow"
                        st["ready"] = sched_cycle + 1
                    elif phase == "level3_final_cmp":
                        cmp_dest = (
                            tmpb[st["tmpb_slot3"]]
                            if self.reuse_addr_as_tmpa
                            else tmpa[g]
                        )
                        if not add_simple_vec(
                            "<", cmp_dest, idx[g], level3_split_v
                        ):
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
                        if hi == 2:
                            op2 = "^"
                        if not add_simple_vec(op2, val[g], tmpa[g], hash_tmp):
                            break
                        st["hash_i"] += 2 if hi == 2 else 1
                        if st["hash_i"] < len(HASH_STAGES):
                            st["phase"] = "hash_pre"
                        elif r + 1 >= rounds:
                            st["tmpb_slot"] = None
                            advance_after_update(g, sched_cycle + 1)
                        elif (r + 1) % (forest_height + 1) == 0:
                            st["tmpb_slot"] = None
                            if done_count >= zero_skip_done_threshold:
                                advance_after_update(g, sched_cycle + 1)
                            else:
                                st["phase"] = "zero"
                        else:
                            st["tmpb_slot"] = None
                            st["phase"] = "parity"
                        maybe_add_base_ready(g, st, r)
                        st["ready"] = sched_cycle + 1
                    elif phase == "hash5_combine":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        hash_tmp = tmpb[tmpb_slot]
                        hash5_combine_added = (
                            add_alu_vec("^", val[g], val[g], hash_tmp)
                            if self.bias_hash5_combine_alu
                            else add_simple_vec("^", val[g], val[g], hash_tmp)
                        )
                        if not hash5_combine_added:
                            break
                        st["tmpb_slot"] = None
                        if r + 1 >= rounds:
                            st["phase"] = "final_unbias"
                        elif (r + 1) % (forest_height + 1) == 0:
                            if done_count >= zero_skip_done_threshold:
                                advance_after_update(g, sched_cycle + 1)
                            else:
                                st["phase"] = "zero"
                        else:
                            st["phase"] = "parity"
                        maybe_add_base_ready(g, st, r)
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
                        maybe_add_base_ready(g, st, r)
                        st["ready"] = sched_cycle + 1
                    elif phase == "hash_pre_second_low4":
                        tmpb_slot = st["tmpb_slot"]
                        if tmpb_slot is None:
                            continue
                        hash_tmp = tmpb[tmpb_slot]
                        hi = st["hash_i"]
                        _op1, _val1, _op2, op3, val3 = HASH_STAGES[hi]
                        if not add_alu_vec_lanes(
                            op3, hash_tmp, val[g], hash_const_v[val3], 0, 4
                        ):
                            break
                        st["phase"] = "hash_combine"
                        maybe_add_base_ready(g, st, r)
                        st["ready"] = sched_cycle + 1
                    elif phase == "parity":
                        if not add_simple_vec("&", tmpa[g], val[g], one_v):
                            break
                        st["phase"] = (
                            "root_add_one"
                            if r % (forest_height + 1) == 0
                            else (
                                (
                                    "select_inc_base_bias"
                                    if self.use_bias_free_c5
                                    else "select_inc_base"
                                )
                                if st["base_ready"]
                                else (
                                    "biased_heap_madd"
                                    if self.use_biased_heap_index
                                    else (
                                        "madd_parity_raw"
                                        if self.use_additive_index_update
                                        else "select_inc"
                                    )
                                )
                            )
                        )
                        st["ready"] = sched_cycle + 1
                    elif phase == "final_unbias":
                        bias_added = (
                            add_alu_vec(
                                "^",
                                val[g],
                                val[g],
                                hash_const_v[0xB55A4F09],
                            )
                            if self.bias_c5_edge_alu
                            else add_simple_vec(
                                "^",
                                val[g],
                                val[g],
                                hash_const_v[0xB55A4F09],
                            )
                        )
                        if not bias_added:
                            break
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "root_add_one":
                        root_base = (
                            two_v
                            if self.use_biased_heap_index
                            else root_child_base_v
                        )
                        root_op = "-" if self.use_bias_free_c5 else "+"
                        root_a = root_base if self.use_bias_free_c5 else tmpa[g]
                        root_b = tmpa[g] if self.use_bias_free_c5 else root_base
                        if not add_simple_vec(root_op, idx[g], root_a, root_b):
                            break
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "select_inc_base":
                        if not add_simple_vec("+", idx[g], addr[g], tmpa[g]):
                            break
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "select_inc_base_bias":
                        if not add_simple_vec("-", idx[g], addr[g], tmpa[g]):
                            break
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "madd_parity_raw":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", addr[g], idx[g], two_v, tmpa[g])
                        )
                        st["phase"] = "add_index_base"
                        st["ready"] = sched_cycle + 1
                    elif phase == "biased_heap_madd":
                        if len(valu_slots) >= SLOT_LIMITS["valu"]:
                            break
                        valu_slots.append(
                            ("multiply_add", idx[g], idx[g], two_v, tmpa[g])
                        )
                        advance_after_update(g, sched_cycle + 1)
                    elif phase == "add_index_base":
                        if not add_simple_vec(
                            "+", idx[g], addr[g], addr_update_const_v
                        ):
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

                if self.level3_prep_alu_hybrid:
                    for g in flow_scan_order:
                        st = states[g]
                        prep_state = st["l3_prep_state"]
                        if (
                            not isinstance(prep_state, int)
                            or prep_state not in (0, 1, 2, 3, 5, 6, 7, 8)
                            or st["l3_prep_ready"] > sched_cycle
                        ):
                            continue
                        pair_slot = st["tmpb_slot2"]
                        selected_slot = st["tmpb_slot4"]
                        work_slot = st["tmpb_slot5"]
                        if None in (pair_slot, selected_slot, work_slot):
                            continue
                        pair = tmpb[pair_slot]
                        selected = tmpb[selected_slot]
                        work = tmpb[work_slot]
                        prep_alu_ops = {
                            0: (
                                "*",
                                selected,
                                pair,
                                level3_base_pair_delta[0],
                            ),
                            1: ("+", selected, selected, level3_base_v[0]),
                            2: ("*", work, pair, level3_base_pair_delta[1]),
                            3: ("+", work, work, level3_base_v[2]),
                            5: (
                                "*",
                                work,
                                pair,
                                level3_diff_pair_delta[0],
                            ),
                            6: ("+", work, work, level3_diff_v[0]),
                            7: (
                                "*",
                                pair,
                                pair,
                                level3_diff_pair_delta[1],
                            ),
                            8: ("+", pair, pair, level3_diff_v[2]),
                        }
                        if not add_alu_vec(*prep_alu_ops[prep_state]):
                            continue
                        st["l3_prep_state"] = prep_state + 1
                        st["l3_prep_ready"] = sched_cycle + 1
                        break

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
        self.optimize_schedule()

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
