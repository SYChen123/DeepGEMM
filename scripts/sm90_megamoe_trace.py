"""Decode the SM90 fused MegaMoE device buffer as a Chrome/Perfetto timeline."""

TRACE_SLOTS = 2048
TRACE_ROLES = 4
CYCLE_SLOTS = 6
ROLE_NAMES = ("Dispatch warp 0", "A/SFA TMA warp", "Math warpgroup 0", "Math warpgroup 1")
EVENT_NAMES = {
    1: "CTA start", 2: "Local routing done", 3: "Pull done", 4: "Pull starts",
    5: "Workspace cleanup starts", 6: "Ring slot wait starts", 7: "Ring slot ready",
    8: "Remote routing counts ready", 9: "Pull arrivals published",
    10: "Receive counts issued", 11: "Receive counts published", 12: "Source flag seen",
    19: "Tiles done", 20: "Combine starts", 21: "Combine done", 22: "Dispatch done",
    23: "Prologue done", 25: "Wait-combine done",
    30: "Activation wait starts", 31: "Activation ready", 32: "Last A/SFA TMA issued",
    38: "Math clock end", 39: "Math clock start", 40: "Tile starts", 41: "Weight scale ready",
    42: "First stage ready", 43: "Mainloop done", 44: "Epilogue done", 45: "L2 ring slot released",
    48: "Weight scale peek", 50: "SwiGLU math done", 51: "L1 output staged",
    52: "L1 store issued", 53: "L1 store drained", 54: "L1 arrival publication",
    60: "L2 BF16 conversion done", 61: "L2 scatter issued", 62: "L2 scatter sync done",
    71: "Early combine expert published", 72: "Early combine counts",
}


def unpack(track, cta, role):
    footer_count, footer_present = track[-1]
    limit = TRACE_SLOTS - 1 - (CYCLE_SLOTS if role >= 2 else 0)
    if not footer_present:
        if role >= 2 and not any(tag or ts for tag, ts in track):
            return [], 0, []
        raise RuntimeError(f"CTA {cta} role {role} has no trace footer. Rebuild the #473 wheel.")
    if footer_present != 1 or footer_count < 0:
        raise RuntimeError(f"CTA {cta} role {role} has an invalid trace footer")
    records = []
    prev_ts = 0
    for tag, ts in track[:min(footer_count, limit)]:
        event = tag >> 56
        if event not in EVENT_NAMES or ts <= 0 or ts < prev_ts:
            raise RuntimeError(f"CTA {cta} role {role} has an invalid event at slot {len(records)}")
        records.append((event, tag & ((1 << 56) - 1), ts))
        prev_ts = ts
    cycles = [value for slot in track[limit:-1] for value in slot]
    return records, max(footer_count - limit, 0), cycles


def export_trace(buffer, metadata, only_cta=None):
    ctas = range(buffer.size(0)) if only_cta is None else (only_cta,)
    tracks = {
        (cta, role): unpack(buffer[cta, role].tolist(), cta, role)
        for cta in ctas for role in range(TRACE_ROLES)
    }
    timestamps = [ts for records, _, _ in tracks.values() for _, _, ts in records]
    if not timestamps:
        raise RuntimeError("MegaMoE trace contains no timestamps")
    origin = min(timestamps)
    events = []
    truncated = []
    math_cycles = []

    def span(name, start, end, args=None):
        events.append({
            "name": name, "ph": "X", "pid": cta, "tid": role,
            "ts": (start - origin) / 1000, "dur": (end - start) / 1000, "args": args or {},
        })

    for (cta, role), (records, dropped, cycles) in tracks.items():
        if role == 0:
            events.append({"ph": "M", "pid": cta, "tid": 0,
                           "name": "process_name", "args": {"name": f"CTA {cta}"}})
        if dropped:
            truncated.append({"cta": cta, "role": ROLE_NAMES[role], "dropped": dropped})
        if cycles:
            math_cycles.append({
                "cta": cta, "role": ROLE_NAMES[role],
                "w13": cycles[:CYCLE_SLOTS], "w2": cycles[CYCLE_SLOTS:],
            })
        if not records:
            continue
        events.append({"ph": "M", "pid": cta, "tid": role,
                       "name": "thread_name", "args": {"name": ROLE_NAMES[role]}})
        markers = {}
        tile = {}
        for event, aux, ts in records:
            args = {"aux": aux}
            if event == 9:
                args.update(rows=aux & 255, token_index=aux >> 8)
            if event in (30, 31, 32):
                args.update(tile=aux & 0xffffffff, linear="W13" if aux >> 32 == 1 else "W2")
            if event == 40:
                tile = {
                    "linear": "W13" if aux & 255 == 1 else "W2",
                    "expert": (aux >> 8) & 65535, "m_block": (aux >> 24) & 255,
                    "n_block": (aux >> 32) & 255, "tile": aux >> 40,
                }
                for mark in (41, 42, 43):
                    markers.pop(mark, None)
            if role >= 2 and 40 <= event <= 62:
                args.update(tile)
            events.append({
                "name": EVENT_NAMES[event], "ph": "i", "s": "t", "pid": cta, "tid": role,
                "ts": (ts - origin) / 1000, "args": args,
            })
            markers[event] = ts
            if role == 0:
                boundaries = {23: (1, "Prologue / PDL wait"), 4: (23, "Routing and metadata"),
                              3: (4, "Activation pull"), 22: (5, "Workspace cleanup")}
                if event in boundaries:
                    start, name = boundaries[event]
                    if start in markers:
                        span(name, markers[start], ts)
            elif role == 1:
                if event == 31:
                    span(args["linear"] + " activation wait", markers[30], ts, args)
                elif event == 32:
                    span(args["linear"] + " A/SFA issue and stage waits", markers[31], ts, args)
            elif event == 42:
                span(tile["linear"] + " first-stage preparation", markers[40], ts, tile)
            elif event == 43:
                first = markers.get(42, markers[40])
                span(tile["linear"] + " mainloop including waits", first, ts, tile)
            elif event == 44 and 43 in markers:
                span(tile["linear"] + " epilogue", markers[43], ts, args)
            elif event == 21 and 20 in markers:
                span("Combine", markers[20], ts, args)
    return {
        "traceEvents": events, "displayTimeUnit": "ns",
        "metadata": {
            **metadata, "selected_cta": only_cta, "event_span_us": (max(timestamps) - origin) / 1000,
            "truncated_tracks": truncated,
        },
        # These are representative-lane clock cycles accumulated by the device
        # in the mainloops that have cycle marks. The bins vary by specialization.
        "mainloop_cycle_bins": [
            "full_wait", "fp4_decode", "other", "rs_wgmma_wait",
            "rs_promote_and_release", "rs_extra",
        ],
        "mainloop_cycles": math_cycles,
    }
