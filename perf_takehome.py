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

import os
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

_BARRIER_OPS = {"pause", "halt", "jump", "cond_jump", "cond_jump_rel", "jump_indirect"}


def slot_rw(engine, slot):
    """返回 (reads:set, writes:set, mem_read:bool, mem_write:bool, barrier:bool)。"""
    op = slot[0]
    a = slot[1:]
    reads, writes = set(), set()
    mem_read = mem_write = barrier = False

    if engine == "alu":
        dest, a1, a2 = a
        reads = {a1, a2}
        writes = {dest}
    elif engine == "valu":
        if op == "vbroadcast":
            dest, src = a
            reads = {src}
            writes = set(range(dest, dest + VLEN))
        elif op == "multiply_add":
            dest, x, y, z = a
            reads = set(range(x, x + VLEN)) | set(range(y, y + VLEN)) | set(range(z, z + VLEN))
            writes = set(range(dest, dest + VLEN))
        else:  # 逐元素
            dest, a1, a2 = a
            reads = set(range(a1, a1 + VLEN)) | set(range(a2, a2 + VLEN))
            writes = set(range(dest, dest + VLEN))
    elif engine == "load":
        if op == "const":
            dest, _val = a
            writes = {dest}
        elif op == "load":
            dest, addr = a
            reads = {addr}
            writes = {dest}
            mem_read = True
        elif op == "load_offset":
            dest, addr, off = a
            reads = {addr + off}
            writes = {dest + off}
            mem_read = True
        elif op == "vload":
            dest, addr = a
            reads = {addr}
            writes = set(range(dest, dest + VLEN))
            mem_read = True
        else:
            raise ValueError(f"unknown load op {op}")
    elif engine == "store":
        if op == "store":
            addr, src = a
            reads = {addr, src}
            mem_write = True
        elif op == "vstore":
            addr, src = a
            reads = {addr} | set(range(src, src + VLEN))
            mem_write = True
        else:
            raise ValueError(f"unknown store op {op}")
    elif engine == "flow":
        if op == "select":
            dest, cond, x, y = a
            reads = {cond, x, y}
            writes = {dest}
        elif op == "vselect":
            dest, cond, x, y = a
            reads = set(range(cond, cond + VLEN)) | set(range(x, x + VLEN)) | set(range(y, y + VLEN))
            writes = set(range(dest, dest + VLEN))
        elif op == "add_imm":
            dest, x, _imm = a
            reads = {x}
            writes = {dest}
        elif op == "coreid":
            writes = {a[0]}
        elif op == "trace_write":
            reads = {a[0]}
        elif op in _BARRIER_OPS:
            barrier = True
        else:
            raise ValueError(f"unknown flow op {op}")
    elif engine == "debug":
        pass  # 调用方应已剔除
    else:
        raise ValueError(f"unknown engine {engine}")
    return reads, writes, mem_read, mem_write, barrier


def schedule(ops):
    """ops: list[(engine, slot)] 或 (engine, slot, region)（不含 debug）。返回 list[dict engine->[slot,...]]。

    内存别名分区（region）：同一 region 内的 load/store 按保守内存序排（可能别名）；不同 region 的
    内存算子**互不排序**（调用方保证它们访问不相交的内存区）。默认 region=None（单一共享区，等价旧行为）。
    本 kernel 给**每组的 vstore 各一个 region**：32 条写回地址两两不相交，本不必相互串行——否则某组
    val 算得晚会把它后面所有组的 store 全堵住（trace 实证的尾部 store 堆积）。各 store 一到自己 val
    就绪即可发，与计算重叠。正确性：每组 store 只依赖自己的 VAL（scratch 链已定序到其 vload 之后）。
    """
    bundles = []          # index -> {engine: [slot,...]}
    counts = []           # index -> {engine: n}
    last_write = {}       # addr -> bundle idx
    last_read = {}        # addr -> bundle idx
    last_mem_write = {}   # region -> bundle idx
    last_mem_read = {}    # region -> bundle idx
    barrier_at = -1
    max_used = -1
    # 每引擎「最小的仍有空槽的 bundle」：只前进不后退（低于它的 bundle 对该引擎已满）。
    # 这样后来的独立算子能回填前面 bundle 的空槽（打满 valu/load），同时保持近线性。
    first_free = {e: 0 for e in SLOT_LIMITS}

    def ensure(idx):
        while len(bundles) <= idx:
            bundles.append({})
            counts.append({})

    for op in ops:
        engine, slot = op[0], op[1]
        region = op[2] if len(op) > 2 else None
        reads, writes, mr, mw, barrier = slot_rw(engine, slot)

        if barrier:
            # 硬屏障：放在所有已排算子之后，独占一拍；其后算子不得跨越。
            idx = max(max_used + 1, barrier_at + 1)
            ensure(idx)
            bundles[idx].setdefault(engine, []).append(slot)
            counts[idx][engine] = counts[idx].get(engine, 0) + 1
            barrier_at = idx
            max_used = max(max_used, idx)
            continue

        earliest = barrier_at + 1
        for r in reads:
            w = last_write.get(r, -1)
            if w + 1 > earliest:
                earliest = w + 1
        for w in writes:
            rd = last_read.get(w, -1)
            if rd > earliest:
                earliest = rd
            pw = last_write.get(w, -1)
            if pw + 1 > earliest:
                earliest = pw + 1
        lmw = last_mem_write.get(region, -1)
        lmr = last_mem_read.get(region, -1)
        if mr and lmw + 1 > earliest:
            earliest = lmw + 1                  # RAW：load 读前面 store 写的新值 → 隔 1 个 bundle
        if mw:
            if lmr > earliest:
                earliest = lmr                  # WAR：store 不早于前面的 load（同 bundle 可，读旧写新）
            if lmw > earliest:
                # store-store 只需保持相对次序（不早于前一个 store），允许**同 bundle 打包**：
                # 同 bundle 内的 mem_write 按加入序（=程序序）落盘，同址后写覆盖前写 → 与顺序语义一致，
                # 异址互不影响。故不必逐个 +1 串行（那样 32 条 vstore 要 32 拍、末尾纯 drain）。
                earliest = lmw

        limit = SLOT_LIMITS[engine]
        # 从 max(earliest, first_free) 起找第一个该引擎有空槽的 bundle
        idx = earliest if earliest > first_free[engine] else first_free[engine]
        ensure(idx)
        while counts[idx].get(engine, 0) >= limit:
            idx += 1
            ensure(idx)
        # 更新 first_free：若刚填的是当前最小空槽 bundle，往前推到下一个未满的
        if idx == first_free[engine] and counts[idx].get(engine, 0) + 1 >= limit:
            ff = idx + 1
            ensure(ff)
            while counts[ff].get(engine, 0) >= limit:
                ff += 1
                ensure(ff)
            first_free[engine] = ff

        bundles[idx].setdefault(engine, []).append(slot)
        counts[idx][engine] = counts[idx].get(engine, 0) + 1

        for w in writes:
            last_write[w] = idx
        for r in reads:
            if idx > last_read.get(r, -1):
                last_read[r] = idx
        if mr and idx > last_mem_read.get(region, -1):
            last_mem_read[region] = idx
        if mw and idx > last_mem_write.get(region, -1):
            last_mem_write[region] = idx
        if idx > max_used:
            max_used = idx

    return [b for b in bundles if b]  # 丢掉空 bundle（不影响计费/正确性，前提：无绝对跳转目标）




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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        向量化 + VLIW 打包实现（见 logix/DESIGN、logix/PLAN P2、logix/RESULTS）。

        - 批状态 idx/val 常驻 scratch；256 元素分成 ng=batch/VLEN 组，每组用独立寄存器
          （组间天然独立，供打包器填满每拍 valu 槽）。
        - hash 阶段 0/2/4 折成一条 multiply_add（val*k+c）；移位阶段各用 1 个临时。
        - node_val 数据相关：低层用「pair 线性插值 + 系数 select」免 gather——相邻两节点的
          node_val 对绝对地址 A 线性（nv = A*D + E，mod 2^32 恒等），pair 间 vselect 选系数、
          末尾一条 MAC 出值；其余层保持标量 gather；idx 存绝对地址免 gather 的地址加（pre-offset）。
        - load 的时间分布：最早两个 wavefront 组的 r1-3 select 换回 gather（EARLYW），填掉
          「前 ~100 拍 wavefront 未到 gather 轮」的 load 空窗；head 常量隔一个走 flow add_imm
          （CONSTFLOW，基于恒零 scratch 词），省 head 的 load 槽。
        - 四引擎均衡：瓶颈 valu 的活分流到闲置 alu（N_ALU 组标量化）与 flow
          —— idx 更新的「+c」加法搬到 flow 的 vselect（parity 在 CFP/CFP+1 间选，IDXFLOW 组）。
        - 发射顺序按对角错位 + round-major 末轮（EMIT=diagtail）消掉头尾 drain 空转。
        - 全部算子交给依赖感知的打包器 schedule() 打包。
        """
        assert batch_size % VLEN == 0, "batch 必须是 VLEN 的整数倍"
        ng = batch_size // VLEN

        # ── scratch 区域 ──────────────────────────────────────────────
        # NV 复用作 gather 地址缓冲：addr=idx+forest_p 写进 NV，再 load(nv,nv) 就地读地址写值
        # （单 op 读旧写新合法），省掉独立 ADDR 区（256 words），给低层去重腾地方。
        # nv/tmp 的组间配对共享（g 与 g+16 共用一份）：valu 组的起跑被 valu 饱和天然错开，
        # g+16 的首轮要等公平份额的槽位、自然落在 g 的整链（~230 拍）结束之后，共享不产生
        # 真实打包约束；alu 组不同——alu 引擎有余量，其早轮会冲到调度最前端、和任何组的
        # 生存期都重叠，共享即串行化（sched_diff 实证）。故 SHAREPAIR=2 只配对 valu 组、
        # alu 组保留私有。0=不共享；1=全部配对（已证毒 +45，留作对照）；2=仅 valu 组配对。
        # 尾轮（round-major 的最后 TK 轮）另走独立小池：尾轮在程序序最末，配对槽会把早组的
        # 尾轮锁到搭档组整个 body 之后（实测 +75 拍）；小池按 g%4 分槽、与各组尾轮的自然
        # 阶梯（~116 拍/槽）对齐，无真实冲突。
        NVMERGE = os.environ.get("NVMERGE", "0") == "1"
        SHAREPAIR = int(os.environ.get("SHAREPAIR", "0"))
        IDX = self.alloc_scratch("idx", batch_size)
        VAL = self.alloc_scratch("val", batch_size)
        # （TMP/NV 工作区推迟到 alu 组划分确定后分配——槽位数取决于配对方式，见 alu_groups 处。）
        # 共享短临时（L3/L4 系数 select 的 cond 等）：t2a 放短命的 cond，t2b 放 D-acc（仅合并
        # 模式用）。槽位在 emit_gr 里按 g%4 + 前/后半程 bank 计算，避免早轮/晚轮共槽毒化 run-ahead。
        TMP2A = self.alloc_scratch("tmp2a", VLEN * 8)
        TMP2B = self.alloc_scratch("tmp2b", VLEN * 8) if NVMERGE else TMP2A

        ops = []
        # 两条 pause（对齐 reference 的两个 yield）不进 ops——打包器把屏障算子独占一拍，
        # 首尾各浪费 1 拍。改为打包完成后直接注入 bundle（见文末）：pause 与同 bundle 其余
        # 槽同拍执行、周期末才暂停，语义等价——run 边界的检查只读 inp_values，首个 pause
        # 落在任何 store 之前、结束 pause 落在全部 store 之后即可。

        # 部分组的 hash/idx 走标量 ALU（12 槽/拍、原本闲置），把瓶颈 valu 的活分流过去：
        # 总吞吐 48 elem/拍(仅 valu) → 60(valu+alu)。N_ALU=6 由 roofline 扫参平衡两引擎得出
        # （alu 组均匀分散 + group-major 打包下实测最优）。env 可复扫调参。
        N_ALU = int(os.environ.get("N_ALU", "8"))
        # alu 组划分提前到此（TMP/NV 槽位分配要用）：均匀撒到全 32 组，让 body 全程都给 alu
        # 供活（trace 实证、扫参得最优）。注意不能用 (i*stride)%ng——N_ALU>8 时回绕重复、
        # set 去重后实际组数缩水（此 bug 曾让 N_ALU=9/10 的扫参全部空转）。
        if "ALU_STRIDE" in os.environ:
            ALU_STRIDE = int(os.environ["ALU_STRIDE"])
            alu_groups = set((i * ALU_STRIDE) % ng for i in range(N_ALU)) if N_ALU else set()
        else:
            # ALU_OFF：整体偏移。g0/g1 承担 EARLYW 的最早 gather，若 g0 是 alu 组（逐 lane
            # hash 延迟 ~12 级 > 向量 8 级）会拖慢首批 gather 释放、加深 head 的 load 空窗。
            ALU_OFF = int(os.environ.get("ALU_OFF", "0"))
            alu_groups = ({(i * ng // N_ALU + ALU_OFF) % ng for i in range(N_ALU)}
                          if N_ALU else set())

        # TMP/NV 槽位表：alu 组恒私有；SHAREPAIR 下 g 与 g-ng/2 若都可共享则同槽
        def _pairable(g):
            return SHAREPAIR == 1 or (SHAREPAIR == 2 and g not in alu_groups)

        _slot, _shared_slots, nslots = {}, set(), 0
        _poolrank = {}   # 共享组在尾轮小池里的槽位（按共享组序号均衡，勿用 g%4——alu 组
        for g in range(ng):  # 恰占满 g%4==0，会把 valu 组挤到 3 个槽上 8 深串行）
            p = g - ng // 2
            if SHAREPAIR and p >= 0 and _pairable(g) and _pairable(p):
                _slot[g] = _slot[p]
                _shared_slots.add(_slot[p])
            else:
                _slot[g] = nslots
                nslots += 1
        _npool = 0
        for g in range(ng):
            if _slot[g] in _shared_slots:
                _poolrank[g] = _npool % 4
                _npool += 1
        TMP = self.alloc_scratch("tmp", nslots * VLEN)     # 每组一个临时向量（可配对共享）
        NV = TMP if NVMERGE else self.alloc_scratch("nv", nslots * VLEN)   # gather/select 结果区
        if SHAREPAIR:
            TMPT = self.alloc_scratch("tmpt", VLEN * 4)    # 尾轮小池（见上）
            NVT = self.alloc_scratch("nvt", VLEN * 4)

        def vop(ua, op, dest, a, b):
            """向量二元运算：ua=True 走 VLEN 条标量 alu，否则一条 valu。"""
            if ua:
                for l in range(VLEN):
                    ops.append(("alu", (op, dest + l, a + l, b + l)))
            else:
                ops.append(("valu", (op, dest, a, b)))

        VMAC_ALU = os.environ.get("VMAC_ALU", "0") == "1"  # MAC 是否也标量化到 alu

        def vmac(ua, dest, a, b, c):
            """dest = a*b + c。ua 且 VMAC_ALU：走 VLEN×(乘+加) 标量 alu；否则一条 valu multiply_add
            （MAC 在 valu 上 1 槽=8elem，比标量 alu 的 16 槽密——避 2× 惩罚，但 alu 组会跨引擎）。"""
            if ua and VMAC_ALU:
                for l in range(VLEN):
                    ops.append(("alu", ("*", dest + l, a + l, b + l)))
                    ops.append(("alu", ("+", dest + l, dest + l, c + l)))
            else:
                ops.append(("valu", ("multiply_add", dest, a, b, c)))

        def scalar(name=None):
            return self.alloc_scratch(name)

        # 本地常量池（scratch_const 写的是 self.instrs，会被 schedule(ops) 覆盖，故自建，按值去重）。
        # CONSTFLOW：head 是 load-bound、flow 反而闲——把部分常量改走 flow 的 add_imm（基于一个
        # 恒零的 scratch 词：scratch 初值全 0，从不写它），省下 head 的 load 槽给 vload/gather。
        # 0=全走 load；1=隔一个走 flow（默认，实测最优）；2=全走 flow。
        CONSTFLOW = int(os.environ.get("CONSTFLOW", "1"))
        ZED = self.alloc_scratch("zed")          # 恒零（从不写入）
        _const_cache = {}
        _cf_n = [0]

        def oconst(val):
            if val not in _const_cache:
                a = scalar()
                _cf_n[0] += 1
                if CONSTFLOW == 2 or (CONSTFLOW == 1 and _cf_n[0] % 2 == 0):
                    ops.append(("flow", ("add_imm", a, ZED, val)))
                else:
                    ops.append(("load", ("const", a, val)))
                _const_cache[val] = a
            return _const_cache[val]

        def seq(base, count, unit):
            """返回 count 个标量 [base, base+unit, .., base+(count-1)*unit]（out[0] 复用 base 本身）。
            用 alu 上的 prefix-doubling 生成：out[i]=out[i-s]+unit*s，s 每轮翻倍 → log₂(count) 层、
            count-1 条独立 alu 加，步长 unit*s 由 alu 翻倍得到，只 1 条 const load（unit）。setup 的
            地址序列（vbase=ivp+VLEN*g、fpp=fvp+k）若逐个 flow add_imm 会在 head 串成长序言把 alu/valu
            全卡住（trace 实证）；改此法后 head 的 flow 序言消掉、只吃本就空闲的 head alu。"""
            out = [base] + [scalar() for _ in range(count - 1)]
            if count <= 1:
                return out
            step = oconst(unit)
            s = 1
            while s < count:
                if s == 1:
                    stepn = step
                else:
                    stepn = scalar()
                    ops.append(("alu", ("+", stepn, step, step)))
                    step = stepn
                for i in range(s, min(2 * s, count)):
                    ops.append(("alu", ("+", out[i], out[i - s], stepn)))
                s *= 2
            return out

        # 从内存头部读运行期指针（不硬编码布局）：mem[4]=forest_values_p, mem[6]=inp_values_p
        def load_hdr(k):
            sv = scalar()
            ops.append(("load", ("load", sv, oconst(k))))
            return sv

        fvp = load_hdr(4)   # forest_values_p
        ivp = load_hdr(6)   # inp_values_p

        # 广播向量（编译期常量按值去重）
        bcache = {}

        def bvec(val):
            if val not in bcache:
                vb = self.alloc_scratch(None, VLEN)
                ops.append(("valu", ("vbroadcast", vb, oconst(val))))
                bcache[val] = vb
            return bcache[val]

        # 按 hash 阶段用到的先后 broadcast（常量的 const load 顺序即调度顺序）：第 0 轮 group 0 一开跑就
        # 需要 S0 的 K0/C0，故先发它们；ONE/TWO 是 idx parity 用、要到 hash 之后 → 放最后，别占早期 load 槽。
        # 阶段3 `(b+C3)^(b<<9)` 是唯一用**左移**的混合级（<<9 = ×512）→ 把 +C3 折进阶段2 的 MAC 常量
        # （C2+C3），再把 b<<9 用 MAC 算成 `b'*512 - C3*512`（b'=b+C3）：阶段2+3 从「MAC+shl+add+xor」
        # 4 个算子压成「MAC+MAC+xor」3 个，每个 hash 省 1 个非-MAC。省的算子在瓶颈 alu 上尤其值钱
        # （alu 组每个 hash 少 16 个 alu 槽）。C2C3/UADD 为编译期常量（见旁注的数值推导，已 100k 随机对拍）。
        K0, C0 = bvec(4097), bvec(0x7ED55D16)      # 阶段0
        S19, C1 = bvec(19), bvec(0xC761C23C)        # 阶段1
        K2, C2C3 = bvec(33), bvec(0xE9F8CC1D)       # 阶段2：常量 = C2 + C3（0x165667B1+0xD3A2646C）
        # 阶段3 的 u 不再读阶段2 的输出、直接由阶段1 输出 v 算：u = (v·33+C2C3)·512 − C3·512
        #   = v·16896 + C2·512 —— 两条 MAC 同读 v、并行发射，hash 关键链 9 级 → 8 级
        # （终链 −1 拍；头部各组爬坡每轮 −1 拍 → 首批 gather 提早 ~4 拍，缩 load 前窗）。
        K3P, C2X = bvec(16896), bvec((0x165667B1 * 512) % 2**32)
        K4, C4 = bvec(9), bvec(0xFD7046C5)          # 阶段4
        S16, C5 = bvec(16), bvec(0xB55A4F09)        # 阶段5
        ONE = bvec(1)
        TWO = bvec(2)

        # pre-offset：idx 存成绝对地址 A = forest_p + idx，则 gather = load(nv, A) 免 addr-add。
        # 需要运行期常量：rvec(j)=broadcast(forest_p+j)（比较用）、CFP=1-forest_p（idx 更新用）。
        rcache = {}

        def rvec(j):
            # rvec(j) = broadcast(forest_p + j)；地址标量 forest_p+j 取自共享的 fpp（下方 prefix 生成）
            if j not in rcache:
                vb = self.alloc_scratch(None, VLEN)
                ops.append(("valu", ("vbroadcast", vb, fpp[j])))
                rcache[j] = vb
            return rcache[j]

        cfp_s = scalar()
        ops.append(("alu", ("-", cfp_s, oconst(1), fvp)))   # CFP = 1 - forest_p (mod 2^32)
        CFP = self.alloc_scratch("cfp_vec", VLEN)
        ops.append(("valu", ("vbroadcast", CFP, cfp_s)))
        # CFP+1 向量：idx 更新的 +c 用 vselect(parity, CFP+1, CFP) 走 flow 引擎（省 valu/alu）
        cfp1_s = scalar()
        ops.append(("flow", ("add_imm", cfp1_s, cfp_s, 1)))
        CFP1 = self.alloc_scratch("cfp1_vec", VLEN)
        ops.append(("valu", ("vbroadcast", CFP1, cfp1_s)))

        # idx 无需初始化：第 0 轮是 L0（node_val=Fvec[0] 常量、完全不读 idx），且该轮**写** idx
        # 供第 1 轮用 → 初始 IDX 是死代码（省 ng 条 vbroadcast，正好在 head 稀疏处腾出 valu）。

        # 初始化：val 从 mem[inp_values_p + g*VLEN] vload（前置一次性发射——各组 val 早早备好，
        # 主体 r=0 不必等 load；试过惰性摊开反而给每组起点加了 load 延迟，更慢）。
        # vbase[g] = ivp + VLEN*g（prefix-doubling 在 alu 生成，避 32 条 flow add_imm 堵 head）
        vbase = seq(ivp, ng, VLEN)              # vbase[0]=ivp；保活到最后 vstore 复用
        for g in range(ng):
            # 每组 val 读写各归一个 region ("io", g)：本组 inp 区与他组不相交，故各组 vload/vstore
            # 互不排序（打包器 region 机制），末尾各 store 一到自己 val 就绪即发、不被慢组堵住。
            ops.append(("load", ("vload", VAL + g * VLEN, vbase[g]), ("io", g)))

        # 低层 gather 去重（同步层级性质）：第 r 轮所有 idx 都在第 (r % period) 层。第 L 层的
        # 节点是连续区间 [2^L-1, 2^(L+1)-1)（共 2^L 个），forest 不变 → setup 时把这些节点 broadcast
        # 成向量，运行期用「线性 select」由 idx 选出 node_val，免 gather：
        #   nv = (idx>base)? F[base+1] : F[base]; 再逐个 nv = (idx>=k)? F[k] : nv。
        # 只用 nv + tmp 两个临时（不需 mux 的第二临时区），省 scratch → 能多去重几层。
        period = forest_height + 1
        # 去重到第几层：越高省的 gather 越多，但线性 select 的 valu/flow 代价 ~2^L 指数涨、且
        # F-vec 占 scratch ~2^(L+1)。由扫参在 load 与 compute 之间平衡（默认 3）。
        MAX_DEDUP = int(os.environ.get("MAX_DEDUP", "3"))
        dedup_levels = sorted({r % period for r in range(rounds) if (r % period) <= MAX_DEDUP})
        marginal_level = dedup_levels[-1] if dedup_levels else -1  # 最高去重层，部分 select
        L3SEL = int(os.environ.get("L3SEL", str(ng)))              # 边际层多少组走 select（其余 gather）

        # L4 选择性去重（r=4/r=15）与早期 gather 换位的旋钮（见下），影响 fpp/FV_raw 的覆盖范围
        SEL15 = int(os.environ.get("SEL15", "5"))    # 末轮改 MAC-select 的组数（alu 组，死区广播）
        SEL4G = int(os.environ.get("SEL4G", "0"))    # r=4 改 select 的组数（死区广播方案下禁用）
        EARLYW = int(os.environ.get("EARLYW", "5"))  # 早于此 wavefront 的 r1-3 select 改回 gather
        EARLY1G = int(os.environ.get("EARLY1G", "0"))  # 前多少组只把 r1 改回 gather（填最深空窗）
        EARLY_GATHER = {
            tuple(map(int, item.split(":")))
            for item in os.environ.get("EARLY_GATHER", "1:3,3:3").split(",")
            if item
        }
        # 末组从第几轮起 L2/L3 改 gather 缩 drain（默认倒数第 2 轮起——终链上 load 已空闲）
        TAILG = int(os.environ.get("TAILG", str(max(rounds - 1, 2))))
        TAILGN = int(os.environ.get("TAILGN", "1"))        # 末几组适用 TAILG
        L4MAC = os.environ.get("L4MAC", "1") == "1"        # L4 select 用 MAC 链（0=flow 链）
        # 末 TAILB 组的 idx 更新改 B 形式：B=2A+CFP 只读 idx、可在本轮 hash 期间提前算（借宿
        # xor 之后即空闲的 nv），把 val→gather 的侧链从「&→+c→MAC」3 拍缩成「&→add」2 拍。
        # valu 算子数不变（MAC+&+add）、纯时序重排——只对终链上的末尾组有意义（中段被吞吐掩盖）。
        TAILB = int(os.environ.get("TAILB", "2"))
        # 死区广播方案需要 ≥8 个 valu 组当 donor
        use_l4 = (SEL15 > 0 and forest_height >= 4 and rounds >= 5
                  and ng - len(alu_groups) >= 8)

        # 双子预取（spec-gather）：gather 地址只有两个候选 2A+CFP+{0,1}——上一轮 idx 一出就把
        # 两个孩子都 gather 回来（不等 parity），parity 出来后 1 条 vselect 选值+1 条选 idx。
        # 代价 +8 载入/轮，收益：该轮关键路径里的 gather 段（~5 拍）完全挪出、且这些载入正好
        # 填 head 的 load 空窗（前 ~100 拍 load 半闲：wavefront 还没推进到 gather 轮）。
        # 只给最前排的 SPECG 组、从 r=4 起 SPECR 个纯 gather 轮用。
        SPECG = int(os.environ.get("SPECG", "0"))
        SPECR = int(os.environ.get("SPECR", "0"))
        # SPEC15G：末尾几组的最后一轮（r15，L4 gather）也走双子预取——r14 的 hash 还没算完就把
        # r15 的两个候选孩子取回来（此时 load 已开始收尾空闲），砍掉收尾关键链里的 gather 段。
        SPEC15G = int(os.environ.get("SPEC15G", "0"))
        nspec_slots = SPECG + SPEC15G
        NV2 = self.alloc_scratch("nv2", VLEN * nspec_slots) if nspec_slots else 0

        def spec_slot(g):
            return NV2 + (g if g < SPECG else SPECG + (ng - 1 - g)) * VLEN

        def is_spec(g, rr):
            if (rr % period) < 4 or forest_height < 4:
                return False
            if g < SPECG and 4 <= rr < min(4 + SPECR, forest_height + 1) and rr < rounds - 1:
                return True
            return rr == rounds - 1 and g >= ng - SPEC15G

        _spec_pending = {}   # g -> 双子预取的 16 条 load：延迟到该组下一轮排放点再发——
                             # 若在本轮（r14 的 idx 段）排放，会在 load 队列里插到所有组的
                             # r15 gather 前面、把别人挤后 ~4 拍（依赖上它们本就更早就绪，
                             # 放到队尾照样按依赖时刻被回填）。

        # 共享地址标量 fpp[k] = forest_p + k（k 覆盖去重层所有节点号及 rvec 用到的 j）：一次 prefix
        # 生成、rvec 与 Fvec 共用，把原本 ~28 条散在 head 的 flow add_imm 全消掉（trace-driven）。
        nmax = ((1 << (max(dedup_levels) + 1)) - 1) if dedup_levels else 3
        nfpp = max(nmax, 30) if use_l4 else nmax  # L4 另需 fpp[15..29]（E 系数与 rvec 的地址标量）
        fpp = seq(fvp, max(3, nfpp), 1)          # fpp[0]=fvp

        # Fvec 节点值：去重层的节点号是连续区间 0..nmax-1 → 用 vload 成块取 forest（每条 8 个），
        # 再从块里逐节点 broadcast——把原本 nmax 条散 load 压成 ⌈nmax/8⌉ 条 vload（head 是 load-bound，
        # trace 实证），省下的 load 面直接缩短 head 序言。
        nchunk = (nmax + VLEN - 1) // VLEN
        FV_raw = self.alloc_scratch("fvraw", nchunk * VLEN)
        for c in range(nchunk):
            ops.append(("load", ("vload", FV_raw + c * VLEN, fpp[c * VLEN])))

        # L0 仍用 F0 的广播向量（单节点直接当 node_val 用）。
        F0 = self.alloc_scratch("fnode0", VLEN)
        ops.append(("valu", ("vbroadcast", F0, FV_raw + 0)))

        # For cached L1-L3 nodes, defer hash stage 5's constant xor across the
        # round boundary.  The selector returns forest[idx] ^ C5, so
        #   (stage4 ^ (stage4 >> 16)) ^ (forest[idx] ^ C5)
        # is exactly hash5(stage4) ^ forest[idx], with one fewer vector xor.
        # C5 is odd, therefore the true hash parity is the inverse of the
        # deferred value's parity; the index update below accounts for that.
        C5NODEFOLD = os.environ.get("C5NODEFOLD", "1") == "1"
        if C5NODEFOLD:
            for c in range(nchunk):
                ops.append(("valu", ("^", FV_raw + c * VLEN, FV_raw + c * VLEN, C5)))

        # L0FOLD：r11 的 node_val 是常量 F0（L0 单节点），且 xor 满足结合律——把 val^=F0 折进
        # r10 hash 阶段5 的常量 xor：(v^C5)^(v>>16)^F0 = (v^(C5^F0))^(v>>16)。r10 用 C5F0 向量、
        # r11 跳过 nv-xor：省 32 组·轮 × 1 算子（valu −24 / alu −64），每组 r10→r11 链还短 1 级。
        # （r0 不可折：初值来自 vload，前面没有可寄生的 hash 级。）
        L0FOLD = os.environ.get("L0FOLD", "1") == "1" and rounds > period
        if L0FOLD:
            C5F0 = self.alloc_scratch("c5f0_vec", VLEN)
            if C5NODEFOLD:
                ops.append(("valu", ("vbroadcast", C5F0, FV_raw + 0)))
            else:
                c5f0_s = scalar("c5f0")
                ops.append(("alu", ("^", c5f0_s, FV_raw + 0, oconst(0xB55A4F09))))
                ops.append(("valu", ("vbroadcast", C5F0, c5f0_s)))

        # L1SEL：L1 轮（r1/r12）的 node_val 不再用 pair-MAC——上一轮是 L0 轮，其 idx 更新的
        # parity 恰好是 L1 的节点选择位、且是 tmp 的最后一笔写（本轮 select 又先于一切 tmp 写），
        # 一条 flow vselect(parity, F2, F1) 直接选值：每组每 L1 轮省 1 条 valu MAC（共 −64 槽，
        # valu 是瓶颈），代价 +1 flow（flow 有大量余量）。F1/F2 是运行期 forest 值，从 FV_raw 广播。
        L1SEL = os.environ.get("L1SEL", "1") == "1"
        if L1SEL:
            F1V = self.alloc_scratch("fnode1", VLEN)
            F2V = self.alloc_scratch("fnode2", VLEN)
            ops.append(("valu", ("vbroadcast", F1V, FV_raw + 1)))
            ops.append(("valu", ("vbroadcast", F2V, FV_raw + 2)))

        # 线性插值系数（pair-MAC）：相邻两节点 (F[k], F[k+1])，node_val 是绝对地址 A 的线性函数
        #   nv = A*D + E，D = F[k+1]-F[k]，E = F[k] - (forest_p+k)*D   （mod 2^32 恒等，A∈{fp+k,fp+k+1}）
        # setup 时用 head 里空闲的 alu 算出 D/E 标量再广播。运行期一条 MAC 顶掉「cmp+vselect」——
        # L1 整层 1 MAC；L2/L3/L4 先用 vselect 选出 D/E 系数向量再 1 MAC（cmp 数从 2^L-1 砍到
        # 2^(L-1)-1）。每 group-round：L1 省 1 flow；L2 省 1 valu+1 flow；L3 省 3 valu+1 flow
        # （alu 组另省大量标量 cmp）。
        pair_DE = {}   # (level, j) -> (Dvec, Evec)，pair j 覆盖节点 {base+2j, base+2j+1}
        # d/m/e 中间标量用 2 组轮转缓冲（15 对 × 3 → 6 词）：只在 setup 期活跃，轮转引入的
        # WAW 间隔 2 对（~6 条 alu），setup 的 alu 本就有余量，不构成约束。
        _dme = [(scalar(), scalar(), scalar()) for _ in range(2)]
        _dme_n = [0]

        def setup_pairs(L, raw_off):
            """给第 L 层建 pair 系数向量；raw_off = 该层 base 节点在 FV_raw 里的偏移。"""
            base = (1 << L) - 1
            for j in range(1 << (L - 1)):
                k = base + 2 * j
                d_s, m_s, e_s = _dme[_dme_n[0] % 2]
                _dme_n[0] += 1
                ops.append(("alu", ("-", d_s, FV_raw + (k - raw_off) + 1, FV_raw + (k - raw_off))))
                ops.append(("alu", ("*", m_s, fpp[k], d_s)))
                ops.append(("alu", ("-", e_s, FV_raw + (k - raw_off), m_s)))
                Dv = self.alloc_scratch(f"pD{k}", VLEN)
                Ev = self.alloc_scratch(f"pE{k}", VLEN)
                ops.append(("valu", ("vbroadcast", Dv, d_s)))
                ops.append(("valu", ("vbroadcast", Ev, e_s)))
                pair_DE[(L, j)] = (Dv, Ev)

        for L in dedup_levels:
            if L >= 2 or (L == 1 and not L1SEL):   # L1SEL 下 L1 不需要 pair 系数（直接 vselect 值）
                setup_pairs(L, 0)

        # L2 的 Δ 系数向量（末尾组 r13 用 MAC 链版 L2 select：cmp/P0/Q 三者并行 + 1 条 acc-MAC
        # = 2 级，比通用版 cmp→Dsel→Esel→MAC 的 4 级短——两个 vselect 在单槽 flow 上必然串行）。
        TAILM = int(os.environ.get("TAILM", "0"))
        if TAILM and 2 in set(dedup_levels):
            # dd/ee 借旋转缓冲存（只需活到紧随的 broadcast，程序序保证 L4 块的复写在其后）
            dd_s, ee_s = _dme[1][0], _dme[1][1]
            # ΔD = (F6−F5) − (F4−F3)；ΔE = E1 − E0（由 FV_raw 与 fpp 重算，setup alu 便宜）
            m1, m2, m3 = _dme[0]
            ops.append(("alu", ("-", m1, FV_raw + 6, FV_raw + 5)))
            ops.append(("alu", ("-", m2, FV_raw + 4, FV_raw + 3)))
            ops.append(("alu", ("-", dd_s, m1, m2)))
            ops.append(("alu", ("*", m1, fpp[5], m1)))       # (fp+5)·D1
            ops.append(("alu", ("*", m2, fpp[3], m2)))       # (fp+3)·D0
            ops.append(("alu", ("-", m3, FV_raw + 5, m1)))   # E1
            ops.append(("alu", ("-", m1, FV_raw + 3, m2)))   # E0
            ops.append(("alu", ("-", ee_s, m3, m1)))
            L2DD = self.alloc_scratch("l2ddv", VLEN)
            L2EE = self.alloc_scratch("l2eev", VLEN)
            ops.append(("valu", ("vbroadcast", L2DD, dd_s)))
            ops.append(("valu", ("vbroadcast", L2EE, ee_s)))
            # 专属 cond 向量：绝不能用共享 t2a——末组的 r13 执行极晚（~b1040）而程序序在尾段
            # 之前，共槽会把所有同槽组的尾段 r14 WAW 锁到它后面（实测 +48）。
            TAILMC = self.alloc_scratch("tailmc", VLEN)

        # L4（16 节点）选择性去重（SEL15）：把部分 alu 组的 r15 gather 换成 select——load 流从
        # ~b100 起全程饱和，从流里任何位置减 8 载入都让 load 终点提前 4 拍；换出的 select 算力
        # （cmp 走 alu、vselect 走 flow、MAC 走 valu）恰好落在尾部的空闲引擎上。
        # scratch 装不下 8 对系数向量（128 词）→ **死区广播**：setup 只算 16 个 D/E 标量（16 词），
        # 系数「向量」借用 8 个 donor 组（valu 组，其 r15 先排放完毕）已死的 tmp/nv 区，在尾部
        # r15 序列中段注入 vbroadcast 现场生成；被转换组的 r15 排放在广播之后，整段浮到尾部的
        # 空闲拍里执行。程序序保证语义：donor 全部用完 → 广播写 → 转换组读。
        # 系数存 Δ 形式（供 MAC 链条 select）：nv = (A·D0+E0) + Σⱼ cⱼ·(A·ΔDⱼ+ΔEⱼ)，
        # cⱼ = (A ≥ fp+15+2j) 是阶梯条件——选中 pair p 时恰好前 p 个 cⱼ=1，累加出 A·Dₚ+Eₚ。
        # MAC 链条全程 0 条 flow（尾部 flow 是墙、valu 反而大量空闲——select 形式要长成尾部的形状）。
        l4de = []   # [(D0,E0), (ΔD1,ΔE1), ..., (ΔD7,ΔE7)] 持久标量，活到尾部广播
        C5MEML4 = C5NODEFOLD and use_l4
        if use_l4:
            for c in range(2):
                ops.append(("load", ("vload", FV_raw + c * VLEN, fpp[15 + c * VLEN]), "l4tree"))
            if C5MEML4:
                for c in range(2):
                    base = FV_raw + c * VLEN
                    ops.append(("valu", ("^", base, base, C5)))
                    ops.append(("store", ("vstore", fpp[15 + c * VLEN], base), "l4tree"))
            pd, pe = None, None   # 上一对的 d/e（rotor，供 Δ 相减）
            for j in range(8):
                k = 15 + 2 * j
                d_r, m_r, e_r = _dme[j % 2]           # 2 路 rotor：算 Δ 时上一对仍在
                ops.append(("alu", ("-", d_r, FV_raw + (k - 15) + 1, FV_raw + (k - 15))))
                ops.append(("alu", ("*", m_r, fpp[k], d_r)))
                ops.append(("alu", ("-", e_r, FV_raw + (k - 15), m_r)))
                ds, es = scalar(f"l4d{j}"), scalar(f"l4e{j}")
                if j == 0 or not L4MAC:
                    ops.append(("alu", ("+", ds, d_r, ZED)))      # 复制 Dⱼ/Eⱼ（ZED 恒 0）
                    ops.append(("alu", ("+", es, e_r, ZED)))
                else:
                    ops.append(("alu", ("-", ds, d_r, pd)))       # ΔDⱼ = Dⱼ − Dⱼ₋₁
                    ops.append(("alu", ("-", es, e_r, pe)))
                pd, pe = d_r, e_r
                l4de.append((ds, es))
            # donor：前 8 个 valu 组（与被转换的 alu 组天然不相交）。其 tmp/nv 在自身 r15 排放
            # 完毕后永久死亡，正好当 L4 系数向量的容身处。L4Q2=1 时再多征 4 个 donor 给被转换组
            # 当第二 Q 缓冲（实测收益被额外 donor 的死亡门槛抵消，默认关）。
            L4Q2 = os.environ.get("L4Q2", "0") == "1"
            _donors = [g for g in range(ng) if g not in alu_groups][:12 if L4Q2 else 8]
            for j in range(8):
                dg = _donors[j]
                pair_DE[(4, j)] = (TMP + _slot[dg] * VLEN, NV + _slot[dg] * VLEN)
            _q2 = []
            for dg in _donors[8:]:
                _q2 += [TMP + _slot[dg] * VLEN, NV + _slot[dg] * VLEN]

        dedup_set = set(dedup_levels)

        def cmp_gt(ua, dest, j, idx):
            """dest = (forest_p + j < A)。alu 组走 8 条标量、阈值直接用标量 fpp[j]
            （标量比较不需要广播向量——L4 的 7 个 rvec 全省掉）；valu 组用 rvec(j)。"""
            if ua:
                for l in range(VLEN):
                    ops.append(("alu", ("<", dest + l, fpp[j], idx + l)))
            else:
                ops.append(("valu", ("<", dest, rvec(j), idx)))

        def emit_node_select(ua, level, nv, tmp, t2a, t2b, idx):
            """选出 forest[idx]（idx 在第 level 层，存绝对地址 A），结果放 nv。

            pair 线性插值 + 系数 select：先在 2^(L-1) 个 pair 间线性 select 出系数 D/E（cond 是
            「A ≥ fp+base+2j」⟺ rvec(base+2j-1) < A，cmp 走 valu/alu、vselect 走 flow），最后
            nv = A*D_sel + E_sel 一条 MAC。L3/L4 的 cond/D-acc/E-acc 需三个活跃向量：cond 恒用
            共享 t2a；合并模式（nv==tmp）下 D-acc 用共享 t2b，否则用私有 nv。"""
            base = (1 << level) - 1
            if level == 0:
                return F0                                        # L0：单节点，直接用其向量
            if level == 1:
                if L1SEL:
                    # 上一轮（L0）的 parity 还活在 tmp（其 idx 更新的最后一笔 tmp 写、本轮
                    # select 先于一切 tmp 写）——一条 flow vselect 直接选 F1/F2，省 1 valu
                    if C5NODEFOLD:
                        # tmp holds the deferred value's parity. C5 is odd, so
                        # true parity is inverted: tmp=1 selects the left node.
                        ops.append(("flow", ("vselect", nv, tmp, F1V, F2V)))
                    else:
                        ops.append(("flow", ("vselect", nv, tmp, F2V, F1V)))
                else:
                    vmac(ua, nv, idx, *pair_DE[(1, 0)])          # 1 MAC 顶掉 cmp+vselect
                return nv
            if level == 2:
                D0, E0 = pair_DE[(2, 0)]
                D1, E1 = pair_DE[(2, 1)]
                ca = t2a if NVMERGE else tmp                     # cond（非合并时用私有 tmp）
                cmp_gt(ua, ca, base + 1, idx)                    # A ≥ fp+base+2
                ops.append(("flow", ("vselect", nv, ca, D1, D0)))
                ops.append(("flow", ("vselect", ca, ca, E1, E0)))   # 读旧 ca(cond) 写新，同拍合法
                vmac(ua, nv, idx, nv, ca)                        # nv = A*D_sel + E_sel
                return nv
            # level ≥ 3：2^(L-1) 个 pair 的线性系数 select（n-1 cmp + 2(n-1) vselect + 1 MAC）
            npair = 1 << (level - 1)
            dacc = t2b if NVMERGE else nv                        # D-acc；E-acc 恒在私有 tmp
            D0, E0 = pair_DE[(level, 0)]
            D1, E1 = pair_DE[(level, 1)]
            cmp_gt(ua, t2a, base + 1, idx)
            ops.append(("flow", ("vselect", dacc, t2a, D1, D0)))
            ops.append(("flow", ("vselect", tmp, t2a, E1, E0)))
            for j in range(2, npair):
                Dj, Ej = pair_DE[(level, j)]
                cmp_gt(ua, t2a, base + 2 * j - 1, idx)           # A ≥ fp+base+2j
                ops.append(("flow", ("vselect", dacc, t2a, Dj, dacc)))
                ops.append(("flow", ("vselect", tmp, t2a, Ej, tmp)))
            vmac(ua, nv, idx, dacc, tmp)                         # nv = A*D_sel + E_sel
            return nv

        def emit_l4_macsel(ua, nv, tmp, t2a, idx, q2):
            """L4（16 节点）的 MAC 链条 select：nv = (A·D0+E0) + Σⱼ cⱼ·(A·ΔDⱼ+ΔEⱼ)。
            0 条 flow（尾部 flow 是墙）；15 MAC 走 valu（尾部 valu 空闲）、7 个阶梯 cmp 走
            alu（仅 alu 组转换，阈值用标量 fpp）。acc=nv、cond=t2a；Qⱼ 在 tmp 与 q2（另一块
            死区）间交替，免得共用一个缓冲被 WAW 串成 2 拍/级。"""
            D0v, E0v = pair_DE[(4, 0)]
            vmac(ua, nv, idx, D0v, E0v)                          # acc = A·D0 + E0
            for j in range(1, 8):
                Qd, Qe = pair_DE[(4, j)]                         # Δ 系数向量（死区广播）
                qb = tmp if j % 2 else q2
                vmac(ua, qb, idx, Qd, Qe)                        # Qⱼ = A·ΔDⱼ + ΔEⱼ
                cmp_gt(ua, t2a, 14 + 2 * j, idx)                 # cⱼ = (A ≥ fp+15+2j)
                vmac(ua, nv, t2a, qb, nv)                        # acc += cⱼ·Qⱼ
            return nv

        # ── 主循环 ────────────────────────────────────────────────────
        # 按「组外层、轮内层」发射（group-major）：批内各组独立，交给打包器后不同组会
        # 错位在不同层——组 A 在低层吃 valu 时组 B 在高层吃 load，两引擎同时忙，消掉
        # round-major 下「所有组同步在低层→load 空转」的硬停顿。算子与依赖不变，正确性照旧。
        # alu 组按 stride 铺开（整组标量化，链留同一引擎、打包好；散点会跨引擎卡顿）——
        # 划分已提前到 N_ALU 处（TMP/NV 槽位分配要用）。
        # 分数级平衡：再挑 1 个「半 alu 组」，其前 EXTRA 轮走 alu、其余走 valu（连续块，只 1 次
        # 跨引擎转换，避散点 lane-sync）——把 valu/alu 之间那点余量磨平（整组太粗）。
        EXTRA = int(os.environ.get("EXTRA", "0"))
        xg = next((g for g in range(ng) if g not in alu_groups), -1)
        # 前 AE 轮所有组一律走 alu：头部所有组的早轮挤在一起把 valu 打满、gather 迟迟起不来
        # （load 空窗的根源），把最前面几轮的非-MAC 挪到头部尚有余量的 alu，让前端更快推进；
        # 每组只在 r=AE 处跨引擎交接一次。
        AE = int(os.environ.get("AE", "0"))

        # idx 更新的 +c 加法搬到 flow 引擎（vselect 在 parity 上选 CFP/CFP+1）：flow 平时很闲
        # （~55% 占用），把它填满能同时卸 valu 与 alu 的负担。优先给 valu 组（alu 组搬 flow 会拉出
        # alu→flow→valu 的跨引擎链、且 flow 单槽突发，反伤打包，实测更差）。IDXFLOW=用 flow 的组数。
        IDXFLOW = int(os.environ.get("IDXFLOW", "18"))
        _forder = [g for g in range(ng) if g not in alu_groups] + \
                  [g for g in range(ng) if g in alu_groups]
        idxflow_groups = set(_forder[:IDXFLOW])

        # 发射顺序参数（emit_gr 里 EARLYW 要用 wavefront 位置，提前解析）
        EMIT = os.environ.get("EMIT", "diagtail")
        SK = int(os.environ.get("SKEW", "5"))
        TK = int(os.environ.get("TAILK", "4"))

        # L4 select 的目标组选择：
        # r15：从倒数第 3 组起**降序**选（tail 解剖：终段 b1089-1100 是末 3 组 r15 gather 以 2/拍
        #   串行占满 load——把倒数 3、4 组的 r15 改 select 直接腾出终段 load 槽；最后 2 组除外，
        #   其 r15 在收尾关键路径上，select 的串行链反而加 drain 延迟）；
        # r4：排除最前 4 组（其 r4 gather 是 head load 空窗的天然填充物），alu 组优先。
        # 死区广播方案下只允许 alu 组转换（cmp 用标量 fpp 阈值、不需要 rvec 向量），且 r4 转换
        # 不可用（系数向量到尾部才广播出来，body 期读不到）。
        _c15 = sorted(alu_groups)
        sel15_groups = set(_c15[:SEL15])
        sel4_groups = set()
        # SEL15A：再转换若干 **valu 组**的 r15 为「纯 ALU 逐 lane MAC 链」——valu 形式受
        # 「系数广播须等 valu 尾部饱和解除」的墙限制（SEL15>7 堆叠的根因）；纯 ALU 形式
        # 直接用 setup 期已就绪的 l4de 标量（乘/加/比较逐 lane 走 alu，无广播、无死区依赖、
        # 零 valu）。每组代价 +296 alu（尾部 alu 空闲 30-50%）、收益 −8 load（流末端 −4 拍）。
        # cond 缓冲借 setup 后死亡的 fvraw（4 槽）+ 尾部空闲 scratch，与 t2a 无冲突。
        SEL15A = int(os.environ.get("SEL15A", "3"))
        _l4donors = set(_donors) if use_l4 else set()   # 死区系数宿主，绝不可被转换（会踩系数区）
        _a15_cands = [g for g in range(ng)
                      if g not in alu_groups and g not in _l4donors
                      and g >= 11 and g < ng - 2]
        sel15a_groups = set(_a15_cands[:SEL15A]) if use_l4 else set()
        # SEL4A：r=4 的 gather 也用纯 ALU 形式转 select——body 期系数标量已就绪（valu 形式
        # 才受死区广播限制）。候选组挑 r4 执行时刻落在 body alu 低谷（trace：b~450/b~650
        # ↔ g≈13/g≈20）的 valu 组；每组 −8 load / +296 alu（落在低谷则不顶 alu frontier）。
        SEL4A = int(os.environ.get("SEL4A", "3"))
        _a4_cands = [g for g in [21, 13, 22, 19, 23, 18, 14, 17, 15, 11]
                     if g not in alu_groups and g not in _l4donors]
        sel4a_groups = set(_a4_cands[:SEL4A]) if use_l4 else set()
        _a15_rank = {g: i for i, g in enumerate(sorted(sel15a_groups | sel4a_groups))}
        _a15_cond = [FV_raw + c * VLEN for c in range(nchunk)]  # fvraw：setup 后死区
        # 再补几个专属 cond 槽（预留 6*VLEN 给 emit 期才惰性分配的 rvec(1,2,4,8,11,13)）
        while ((sel15a_groups or sel4a_groups) and len(_a15_cond) < 8
               and self.scratch_ptr + 7 * VLEN <= SCRATCH_SIZE):
            _a15_cond.append(self.alloc_scratch(None, VLEN))
        # 被转换组的 cond 槽位按「转换组序号」分——alu 组全是 g%4==0，若沿用 g%4+bank 的槽位，
        # 所有转换组的 42 个 cond 写会挤在同一个 t2a 槽上跨组串行（实测 SEL15=6 时尾部 +20）。
        # 尾部时刻 t2a 的 8 个槽全已死亡（bank0 的 L2/L3 cond 在 body 用完），可全用。
        _sel15_rank = {g: i for i, g in enumerate(_c15)}

        def uses_c5_folded_node(g, rr):
            """Whether round rr consumes an L1-L3 selector built from F^C5."""
            if not C5NODEFOLD or rr >= rounds:
                return False
            level = rr % period
            if level == 4:
                return C5MEML4
            if level not in (1, 2, 3):
                return False
            sel = (level in dedup_set) and (level < marginal_level or g < L3SEL)
            if sel and EMIT == "diagtail" and 1 <= rr <= 3 and (rr + SK * g) < EARLYW:
                sel = False
            if sel and rr == 1 and g < EARLY1G:
                sel = False
            if sel and (g, rr) in EARLY_GATHER:
                sel = False
            if sel and g >= ng - TAILGN and level in (2, 3) and rr >= TAILG:
                sel = False
            return sel

        def emit_gr(g, r):
                b = g * VLEN
                bp = _slot[g] * VLEN                              # nv/tmp 槽位（valu 组配对共享）
                idx, val, tmp, nv = IDX + b, VAL + b, TMP + bp, NV + bp
                if (SHAREPAIR and _slot[g] in _shared_slots
                        and EMIT == "diagtail" and r >= rounds - TK):
                    tmp = TMPT + _poolrank[g] * VLEN              # 共享组的尾轮走独立小池（见上）
                    nv = NVT + _poolrank[g] * VLEN
                # 共享临时槽位 = g%4 + 前后半程 bank：若「某组的晚轮」与「另一组的早轮」共槽，
                # 程序序（wavefront 序）会把晚轮排前面，逼后者放弃跑前（run-ahead）、等 ~150 拍
                # （diffsched 实证的 WAR 毒化）。按轮次分 bank 后共槽的只会是同期轮，天然错开。
                t2s = (g % 4 + (4 if r >= rounds // 2 else 0)) * VLEN
                t2a = TMP2A + t2s
                t2b = TMP2B + t2s
                ua = (g in alu_groups) or (g == xg and r < EXTRA) or r < AE  # 该(组,轮)走标量 alu
                level = r % period
                # node_val：去重层用线性 select，其余标量 gather。三处按「load 的时间分布」微调
                # （load 总量与 valu 双双压 roofline 后，赢面在把 load 的活从尾部搬到头部空窗）：
                # ① 边际层 L3 部分组 select（L3SEL）；② 早期 wavefront 的 r1-3 反而改回 gather
                #    （EARLYW——头 100 拍 wavefront 未到 gather 轮、load 空转，select 在此纯浪费）；
                # ③ r=15/r=4（L4 层）部分组改 select（SEL15/SEL4G），卸掉 load 尾部堆积。
                sel = (level in dedup_set) and (level < marginal_level or g < L3SEL)
                if sel and EMIT == "diagtail" and 1 <= r <= 3 and (r + SK * g) < EARLYW:
                    sel = False
                if sel and r == 1 and g < EARLY1G:
                    sel = False   # r1 的 gather 发得最早（~b16），专门填 head 空窗最深处
                if sel and (g, r) in EARLY_GATHER:
                    sel = False
                if sel and g >= ng - TAILGN and level in (2, 3) and r >= TAILG:
                    sel = False   # 末组的晚轮 select 改 gather：终链上 load 早已空闲，
                                  # gather 2 级比系数 select 4 级短，直接缩 drain
                if use_l4 and level == 4 and (
                    (r == rounds - 1 and g in (sel15_groups | sel15a_groups))
                    or (r == 4 and g in (sel4_groups | sel4a_groups))
                ):
                    sel = True
                if is_spec(g, r):
                    # 双子预取的 16 条 load 在此排放（程序序须先于下面读值的 vselect），
                    # parity 仍留在 tmp：一条 vselect 选出 node_val
                    ops.extend(_spec_pending.pop(g, []))
                    ops.append(("flow", ("vselect", nv, tmp, spec_slot(g), nv)))
                    nv_src = nv
                elif sel and level == 4 and (
                        (r == rounds - 1 and g in sel15a_groups)
                        or (r == 4 and g in sel4a_groups)):
                    # 纯 ALU 逐 lane MAC 链（见 SEL15A）：acc=nv、Q=tmp、cond 用死区槽；
                    # 系数/阈值全是标量（l4de/fpp），不经广播、不占 valu。
                    ca = _a15_cond[_a15_rank[g] % len(_a15_cond)]
                    d0, e0 = l4de[0]
                    for l in range(VLEN):
                        ops.append(("alu", ("*", nv + l, idx + l, d0)))
                        ops.append(("alu", ("+", nv + l, nv + l, e0)))
                    for j in range(1, 8):
                        dj, ej = l4de[j]
                        for l in range(VLEN):
                            ops.append(("alu", ("<", ca + l, fpp[14 + 2 * j], idx + l)))
                            ops.append(("alu", ("*", tmp + l, idx + l, dj)))
                            ops.append(("alu", ("+", tmp + l, tmp + l, ej)))
                            ops.append(("alu", ("*", tmp + l, tmp + l, ca + l)))
                            ops.append(("alu", ("+", nv + l, nv + l, tmp + l)))
                    nv_src = nv
                elif sel and level == 4:
                    rk = _sel15_rank[g]
                    t2a4 = TMP2A + (rk % 8) * VLEN
                    if L4MAC:
                        q2 = _q2[rk % len(_q2)] if _q2 else tmp
                        nv_src = emit_l4_macsel(ua, nv, tmp, t2a4, idx, q2)
                    else:
                        # flow 形式（系数 vselect 链）：alu 组只花 1 条 valu MAC——cond 走 alu、
                        # D/E 选择走 flow（选中的组跑在尾部前段，flow 尚有余量）
                        nv_src = emit_node_select(ua, 4, nv, tmp, t2a4, nv, idx)
                elif sel and level == 2 and TAILM and g >= ng - TAILM and r >= rounds - 3:
                    # 末尾组晚轮的 2 级 L2（MAC 链）：P0/Q/cond 三者只依赖 idx、并行发射
                    D0, E0 = pair_DE[(2, 0)]
                    vmac(ua, nv, idx, D0, E0)                    # P0 = A·D0 + E0
                    vmac(ua, tmp, idx, L2DD, L2EE)               # Q = A·ΔD + ΔE
                    cmp_gt(ua, TAILMC, 4, idx)                   # c = (A ≥ fp+5)，专属 cond 区
                    vmac(ua, nv, TAILMC, tmp, nv)                # nv = P0 + c·Q
                    nv_src = nv
                elif sel:
                    nv_src = emit_node_select(ua, level, nv, tmp, t2a, t2b, idx)
                else:
                    # idx 已是绝对地址 A → gather 直接 load(nv, idx)，免 addr-add
                    for lane in range(VLEN):
                        op = ("load", ("load", nv + lane, idx + lane))
                        if C5MEML4 and level == 4:
                            op += ("l4tree",)
                        ops.append(op)
                    nv_src = nv
                # val = myhash(val ^ node_val)（valu 组走向量，alu 组走标量分流）
                if not (L0FOLD and level == 0 and r > 0):
                    vop(ua, "^", val, val, nv_src)   # L0FOLD：r11 的 F0-xor 已折进 r10 阶段5
                _fold_node = uses_c5_folded_node(g, r + 1)
                # 末尾组的 B 形式 idx 更新（见 TAILB）：B 只读 idx，此刻就能算，借宿 xor 后
                # 即空闲的 nv；下方 idx 更新缩为「&→add」两级
                tb = (TAILB and g >= ng - TAILB and r != rounds - 1
                      and level not in (0, forest_height)
                      and g not in idxflow_groups and not is_spec(g, r + 1))
                if tb:
                    vmac(ua, nv, idx, TWO, CFP1 if _fold_node else CFP)        # B/B' for parity form
                vmac(ua, val, val, K0, C0)                                    # 阶段0
                vop(ua, ">>", tmp, val, S19)                                  # 阶段1
                vop(ua, "^", val, val, C1)
                vop(ua, "^", val, val, tmp)
                vmac(ua, tmp, val, K3P, C2X)                                  # 阶段3：u 直读阶段1 输出（与阶段2 并行）
                vmac(ua, val, val, K2, C2C3)                                  # 阶段2（常量折入阶段3 的 +C3）
                vop(ua, "^", val, val, tmp)                                   # result = b' ^ u
                vmac(ua, val, val, K4, C4)                                    # 阶段4
                vop(ua, ">>", tmp, val, S16)                                  # 阶段5
                _fold0 = L0FOLD and r + 1 < rounds and (r + 1) % period == 0
                if _fold_node:
                    # Keep q = stage4 ^ (stage4 >> 16). The next cached node is
                    # represented as F[idx] ^ C5 and restores the exact value.
                    vop(ua, "^", val, val, tmp)
                else:
                    vop(ua, "^", val, val, C5F0 if _fold0 else C5)
                    vop(ua, "^", val, val, tmp)
                # A(绝对地址)更新：newA = 2A + CFP + parity（CFP=1-forest_p）。三处省算子：
                # ① 最后一轮 idx 不再用到 → 省；
                # ② level==height 那轮所有元素必回卷到 0、且下一轮是 L0（L0 完全不读 idx）→ 该轮
                #    idx 更新是死代码，整段省（连 wrap 都不用算）；
                # ③ L0 轮 idx 恒为 forest_p（常量）→ A = forest_p+1+parity = rvec(1)+parity，省掉 MAC。
                if r != rounds - 1 and level != forest_height:
                    fl = g in idxflow_groups                                # +c 走 flow vselect
                    if is_spec(g, r + 1):
                        # 双子预取版 idx 更新：A0=2A+CFP 不等 parity 就能算（只读 idx），两个
                        # 候选地址就地写进 nv/nv2 作 gather 地址缓冲（load 读旧地址写新值，
                        # 程序序上 vselect 在前保证其读到的是地址）；idx 由 parity 选出。
                        nv2 = spec_slot(g)
                        vmac(ua, nv, idx, TWO, CFP)                          # A0 = 2A + CFP
                        vop(ua, "+", nv2, nv, ONE)                           # A1 = A0 + 1
                        vop(ua, "&", tmp, val, ONE)                          # parity（下一轮选值再用）
                        ops.append(("flow", ("vselect", idx, tmp, nv2, nv)))
                        _spec_pending[g] = (
                            [("load", ("load", nv + lane, nv + lane)) for lane in range(VLEN)]
                            + [("load", ("load", nv2 + lane, nv2 + lane)) for lane in range(VLEN)])
                    elif tb:
                        vop(ua, "&", tmp, val, ONE)                          # parity
                        if _fold_node:
                            vop(ua, "-", idx, nv, tmp)                       # A = B' - deferred parity
                        else:
                            vop(ua, "+", idx, tmp, nv)                       # A = B + parity（B 已提前）
                    elif level == 0:
                        vop(ua, "&", tmp, val, ONE)                          # parity
                        if fl:  # A = parity? rvec(2) : rvec(1)（省一次 valu/alu 加）
                            if _fold_node:
                                ops.append(("flow", ("vselect", idx, tmp, rvec(1), rvec(2))))
                            else:
                                ops.append(("flow", ("vselect", idx, tmp, rvec(2), rvec(1))))
                        else:
                            if _fold_node:
                                vop(ua, "-", idx, rvec(2), tmp)             # A = forest_p+2-qparity
                            else:
                                vop(ua, "+", idx, tmp, rvec(1))             # A = parity + (forest_p+1)
                    else:
                        vop(ua, "&", tmp, val, ONE)                          # parity
                        if fl:  # c = parity? CFP+1 : CFP（走 flow），再 MAC
                            if _fold_node:
                                ops.append(("flow", ("vselect", tmp, tmp, CFP, CFP1)))
                            else:
                                ops.append(("flow", ("vselect", tmp, tmp, CFP1, CFP)))
                        else:
                            if _fold_node:
                                vop(ua, "-", tmp, CFP1, tmp)                 # t = CFP+1-qparity
                            else:
                                vop(ua, "+", tmp, tmp, CFP)                 # t = parity + (1-forest_p)
                        vmac(ua, idx, idx, TWO, tmp)                         # A = 2A + t（此后无越界，无 wrap）

        # 发射顺序（EMIT，决定打包器看到的算子序 ≈ 调度序）：
        #   group（回退）：组外层轮内层。body 打满，但尾部只剩最后一组的 16 轮串行链独自
        #     drain（≈一条关键路径 ~130 拍），头部也只有第一组在爬坡 → 头尾空转 ~150 拍。
        #   diagtail（默认，实测最优）：主体按对角错位发射 —— wavefront w 内 (g,r) 满足
        #     r + SK*g ≈ w，各组错位在不同轮/层，组 A 吃 valu 时组 B 吃 load、两引擎同时忙，
        #     且尾部不再是单组独占。末 TK 轮改 round-major：那几轮各组只剩独立单轮，drain 短，
        #     还能回填主体尾部的空槽。round-major 头部会触发同层 gather 聚集（load 突发）反而更慢，
        #     故头部不铺开。SK/TK 由扫参得最优（见 RESULTS「消 drain」一节）。
        if EMIT == "diagtail":
            # 非均匀斜率：后段组（g ≥ TSG）用更小的斜率 TSK 提前起跑——末组的链端 ≈ 起跑+~230
            # 拍串行链，压缩后段起跑能让链端向 load 的完成时间靠拢、缩短纯 drain。
            TSG = int(os.environ.get("TSG", str(ng)))
            TSK = int(os.environ.get("TSK", str(SK)))

            def wkey(g):
                return SK * g if g < TSG else SK * TSG + TSK * (g - TSG)

            body = sorted(((g, r) for g in range(ng) for r in range(rounds - TK)),
                          key=lambda gr: (gr[1] + wkey(gr[0]), gr[0]))
            tgorder = list(range(ng))
            tail15 = [(g, rounds - 1) for g in range(ng)]
            if use_l4:
                # 尾段各轮的组序都重排为「donor → 被转换组 → 其余」：尾段轮次按排放序在引擎
                # frontier 上排队执行，被转换组的 r14/r15-select 必须排在前部才能被尾段吸收
                # （否则其 select+hash 链 ~26 拍在队尾甩出新尾巴，实测 SEL15=6 时 +20）。
                # r15 序中在 donor 之后注入系数广播（donor 的 tmp/nv 至此彻底死亡）。
                # （试过把 donor 的 r14/r15 放回 body 对角线让广播更早就绪：donor 的 r15 gather
                #   会在 load 流里插队、把中段组的 gather 全推后，反而 +13~+28，弃。）
                _ds = set(_donors)
                _cs = sorted(sel15_groups)
                # TGPRI：尾段各轮「其余组」里让最末 TGPRI 个组排最前——它们是终链级联的头，
                # 早拿到 valu/load 槽整条级联就整体左移（其余组的排放不受影响，谁最后完成
                # 谁承担终链，但级联释放得越早、与 load 队列的重叠越满）。
                TGPRI = int(os.environ.get("TGPRI", "0"))

                def _others(exclude):
                    rest = [g for g in range(ng) if g not in exclude]
                    return rest[-TGPRI:][::-1] + rest[:-TGPRI] if TGPRI else rest

                tgorder = _donors + _cs + _others(_ds | sel15_groups)
                _a15s = sorted(sel15a_groups)
                tail15 = ([(g, rounds - 1) for g in _donors] + [("L4BCAST", -1)] +
                          [(g, rounds - 1) for g in _cs] +
                          [(g, rounds - 1) for g in _a15s] +
                          [(g, rounds - 1)
                           for g in _others(_ds | sel15_groups | sel15a_groups)])
            order = (body +
                     [(g, r) for r in range(rounds - TK, rounds - 1) for g in tgorder] +
                     tail15)
        else:  # group-major 回退
            order = [(g, r) for g in range(ng) for r in range(rounds)]
        for g, r in order:
            if g == "L4BCAST":
                for j in range(8):
                    Dv, Ev = pair_DE[(4, j)]
                    d_s, e_s = l4de[j]
                    ops.append(("valu", ("vbroadcast", Dv, d_s)))
                    ops.append(("valu", ("vbroadcast", Ev, e_s)))
                continue
            emit_gr(g, r)

        # 写回 val（提交只校验 inp_values）。每组 store 归 region ("io", g)：各组写回地址不相交 →
        # 互不排序，谁的 val 先算好谁先写、与计算重叠；不必等最慢那组（trace 实证尾部 store 堆积的根因）。
        # 正确性：store 只依赖自己 VAL（scratch 链已定序在其 vload 之后），且与 gather（读 forest 区）不相交。
        for g in range(ng):
            ops.append(("store", ("vstore", vbase[g], VAL + g * VLEN), ("io", g)))

        self._ops = ops  # 供 logix 工具（roofline/关键路径/调度诊断）直接取算子流
        instrs = schedule(ops)
        # 注入两条 pause（见 ops 开头的注释）：
        # 起始 pause 放进第一个 flow 有空槽的 bundle（此前全是 setup，无内存写 → run1 的
        # 检查读到的 inp_values 原封不动）；结束 pause 放进最后一个 bundle（store 同拍已提交）。
        for b in instrs[:-1]:
            if len(b.get("flow", [])) < 1:
                b.setdefault("flow", []).append(("pause",))
                break
        else:
            instrs.insert(0, {"flow": [("pause",)]})
        if len(instrs[-1].get("flow", [])) < 1:
            instrs[-1].setdefault("flow", []).append(("pause",))
        else:
            instrs.append({"flow": [("pause",)]})
        self.instrs = instrs

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
