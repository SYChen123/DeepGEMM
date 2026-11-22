"""Build an offline, cumulative MegaMHC profile comparison without moving raw files.

python scripts/compare_sm90_mhc_profiles.py \
  --input ~/LLMInfer/profile/ncu/sm90-megamhc-decode-20261122 \
  --output ~/LLMInfer/profile/ncu/sm90-megamhc-stages

Requires the verified experiment archive: stages/stages.json, staged-comparison.json
and profiles/{baseline,prefill_opt1,decode_opt1,decode_opt2}/{4096,32}.
The output contains relative symlinks to immutable raw profiles. It never uploads
data, runs a profiler, or modifies the original archive.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics


STAGES = ('baseline', 'prefill_opt1', 'decode_opt1', 'decode_opt2')
DIRECTORIES = ('00_baseline_v1', '01_prefill_opt1_split4',
               '02_decode_opt1_parallel_mix', '03_decode_opt2_split80')
ROLES = ('IO', 'Fn', 'Post0', 'Post1', 'WGMMA', 'Mix', 'Norm')


def summarize_trace(path):
    trace = json.loads(path.read_text())
    meta = trace['metadata']
    assert meta['dropped_events'] == 0, path
    events = [e for e in trace['traceEvents'] if e['ph'] == 'X']
    phases = defaultdict(list)
    for e in events:
        phases[e['name']].append(e)
    summary = {name: {'count': len(items),
                     'median_duration_us': statistics.median(e['dur'] for e in items),
                     'last_end_us': max(e['ts'] + e['dur'] for e in items),
                     'sum_warp_intervals_us': sum(e['dur'] for e in items)}
               for name, items in phases.items()}
    rows = {}
    for mode in ('critical', 'cta0'):
        selected = []
        for role in ROLES:
            candidates = [e for e in events if e['cat'] == role]
            if mode == 'cta0':
                candidates = [e for e in candidates if e['pid'] == 0]
            if not candidates:
                continue
            # Pick one actual warp per role; this is a set of representative
            # timelines, not a reconstructed causal chain across different CTAs.
            preferred = ('Mix affine + Sinkhorn' if role == 'Mix' else
                         'Norm compute + store' if role == 'Norm' else 'projection task')
            useful = [e for e in candidates if e['name'] == preferred] or candidates
            last = max(useful, key=lambda e: e['ts'] + e['dur'])
            warp_events = [dict(name=e['name'], start=e['ts'], duration=e['dur'])
                           for e in candidates if (e['pid'], e['tid']) == (last['pid'], last['tid'])]
            selected.append(dict(role=role, cta=last['pid'], warp=last['tid'], events=warp_events))
        rows[mode] = selected
    return dict(metadata={k: meta[k] for k in ('tokens', 'grid', 'threads', 'trace_schema', 'dropped_events')},
                instrumented_span_us=max(e['ts'] + e['dur'] for e in events),
                phases=summary, rows=rows,
                sinkhorn_iteration_counts=dict(sorted((str(n), sum(x['iterations'] == n for x in meta['sinkhorn_iterations']))
                    for n in set(x['iterations'] for x in meta['sinkhorn_iterations']))))


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SM90 MegaMHC · 累计优化 profile</title>
<style>
:root{color-scheme:light;--ink:#172a3b;--muted:#526b7b;--line:#dce4ea}
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:30px}h1{font-size:28px;margin:4px 0 8px}h2{font-size:18px;margin:0 0 10px}
p{margin:8px 0;color:var(--muted)}.eyebrow{font-size:12px;letter-spacing:.12em;color:#32637f}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}
.card,.panel,.controls{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}
.card strong{display:block;font-size:14px}.card .value{font-size:24px;font-weight:650;margin-top:8px}
.card small{display:block;color:var(--muted)}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}
label{display:flex;align-items:center;gap:8px}select{font:inherit;padding:7px;border:1px solid #b5c8d4;border-radius:6px;background:white;color:var(--ink);max-width:100%}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{min-width:0}.chart{width:100%;display:block}
.legend{display:flex;flex-wrap:wrap;gap:9px 17px;font-size:12px;margin:14px 0}.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}
.metric{font-size:21px;font-weight:650}.links{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}a{color:#136d9c;text-decoration:none}a:hover{text-decoration:underline}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}td,th{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.note{font-size:13px}.positive{color:#087c67}.section{margin-top:20px}.barpanel svg{max-height:175px;min-width:590px}#bars,.timelinewrap{overflow-x:auto}.timelinewrap svg{min-width:670px}
@media(max-width:900px){main{padding:18px}.twocol{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:24px}.controls{gap:12px;align-items:flex-start}label{flex-wrap:wrap}}
@media(max-width:480px){.cards{grid-template-columns:1fr 1fr;gap:8px}.card{padding:12px}.card .value{font-size:21px}.controls{display:grid}label{display:grid;gap:4px}.panel{padding:12px}}
</style><main>
<div class="eyebrow">H200 · 132 SM · H=5120 · HC=4 · SINKHORN=20</div>
<h1>SM90 MegaMHC：逐项对照优化前后</h1>
<p>baseline → prefill opt1 → decode opt1 → decode opt2。所有阶段复用相同真实权重 fixture；原始 profile 保持在归档目录中。</p>
<div class="cards" id="cards"></div>
<div class="controls">
<label>Shape<select id="shape"><option value="32">Decode · 32 tokens/rank</option><option value="4096">Prefill · 4096 tokens/rank</option></select></label>
<label>优化前<select id="before"></select></label><label>优化后<select id="after"></select></label>
<label>观察范围<select id="mode"><option value="critical">各角色最晚完成的代表 warp</option><option value="cta0">CTA 0 的代表 warp</option></select></label>
</div>
<div class="twocol"><section class="panel barpanel"><h2>未打点 CUDA Graph latency</h2><div id="bars"></div></section>
<section class="panel"><h2>选中对比</h2><div class="metric" id="delta"></div><p id="context"></p><p class="note">每阶段两进程、每进程 24 轮、每 graph 64 次调用。顺序 ABCDDCBA；表中为 48 个 samples 的 median。SGLang 使用 decode stats side stream，并在 boundary 末 join。</p></section></div>
<div class="legend" id="legend"></div>
<div class="twocol" id="timelines"></div>
<p class="note">时间线来自单独打点 launch，包含打点开销，横轴采用共同尺度。代表 warp 可来自不同 CTA，不能把它们拼成已证明的因果路径；跨 warp interval 总和也不等于 kernel latency。浅灰是角色生命周期，深色是内部阶段；Post 退休后可参与 Norm。</p>
<section class="panel section"><h2>内部阶段变化</h2><div class="scroll"><table id="phases"></table></div></section>
<section class="panel section"><h2>阶段与原始证据</h2><p>00 原始 MegaMHC V1；01 仅 prefill split 上限目标 16→4；02 decode 一个 WG/token、四 warp 并行汇总；03 在 02 上将 decode producer 40→80 splits。01 的 decode 路径与 00 相同，计时差异视为波动。</p>
<p>全部正式时间包含 barrier clear 与 MegaMHC，不包含 MoE、跨 GPU 通信或模型服务；baseline 名称指 MegaMHC V1，SGLang 是另列的多 kernel 计时参照。</p>
<div class="links"><a href="README.md">完整索引</a><a href="comparison.json">结构化结果</a><a href="evidence/staged-comparison.json">原始计时汇总</a><a href="evidence/manifest.json">原始文件校验清单</a></div></section>
<section class="panel section"><h2>未采用的实验与历史原始 profile</h2><p>有效阶段按上面的累计顺序对照；负收益、微小收益原型保留在实验归档中。</p><div class="links"><a href="experiments/decode_trials/">本轮原型计时</a><a href="evidence/profiles/diagnostic-s80_cycle/32/">Sinkhorn cycle 诊断</a><a href="evidence/profiles/norm16/32/">Norm16 诊断</a><a href="../sm90-megamhc-double-20261121/README.md">V1 / 双 outstanding 原始档</a><a href="../sm90-megamhc-optimization-20261121/README.md">prefill 优化与其他原型</a><a href="../sm90-megamhc-20261118-profiles-v1/README.md">11-18 原始 V1</a></div></section>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const colors={'role':'#dfe7ed','projection task':'#acbed0','Post':'#287ca8','Post IO wait':'#eea33c','Post A wait':'#d27d38','Fn wait':'#ba6d35','A wait':'#dbb353','WGMMA completion gate':'#845db5','Mix dependency wait':'#d2a344','Mix partial reduction':'#d65d66','Mix affine + Sinkhorn':'#bf497f','Mix worker join':'#b38ba5','Norm dependency wait':'#b3be64','Norm compute + store':'#218979'};
function el(tag,attrs={},text){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function html(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
for(const id of ['before','after'])D.stages.forEach((s,i)=>$(id).append(new Option(s.label,i)));$('before').value='0';$('after').value='3';
for(const name of ['Post','Post IO wait','Post A wait','Fn wait','A wait','WGMMA completion gate','Mix dependency wait','Mix partial reduction','Mix affine + Sinkhorn','Norm compute + store']){const e=html('span');const sw=html('span',undefined,'swatch');sw.style.background=colors[name];e.append(sw,document.createTextNode(name));$('legend').append(e)}
function timing(s,t){return D.timing.cases.find(c=>c.tokens===+t).stages.find(x=>x.key===s.key)}
function drawTimeline(s,t,mode,max){const d=s.profiles[t],p=html('section',undefined,'panel');p.append(html('h2',s.label),html('p',`打点跨度 ${d.instrumented_span_us.toFixed(3)} µs · grid ${d.metadata.grid} × 768 · dropped ${d.metadata.dropped_events}`,'note'));
 const svg=el('svg',{viewBox:'0 0 760 343',class:'chart',role:'img','aria-label':s.label+' kernel timeline'}),left=119,w=619,top=30;
 for(let i=0;i<=5;i++){let x=left+w*i/5;svg.append(el('line',{x1:x,x2:x,y1:top,y2:322,stroke:'#e4eaf0'}),el('text',{x,y:18,'text-anchor':'middle','font-size':11,fill:'#526b7b'},(max*i/5).toFixed(1)+' µs'))}
 d.rows[mode].forEach((row,i)=>{let y=top+i*41;svg.append(el('text',{x:0,y:y+15,'font-size':12,fill:'#20394c'},row.role),el('text',{x:0,y:y+30,'font-size':10,fill:'#526b7b'},`CTA ${row.cta} / W${row.warp}`));
  const order={'role':0,'projection task':1};[...row.events].sort((a,b)=>(order[a.name]??2)-(order[b.name]??2)).forEach(e=>{let bg=e.name==='role',task=e.name==='projection task',r=el('rect',{x:left+w*e.start/max,y:y+(bg?8:task?11:15),width:Math.max(.6,w*e.duration/max),height:bg?25:task?19:12,fill:colors[e.name]||'#526b7b',opacity:task ? 0.4 : 1,rx:1});r.append(el('title',{},`${e.name}: ${e.start.toFixed(3)}–${(e.start+e.duration).toFixed(3)} µs (${e.duration.toFixed(3)} µs)`));svg.append(r)})});const wrap=html('div',undefined,'timelinewrap');wrap.append(svg);p.append(wrap);
 const links=html('div',undefined,'links');for(const[file,label]of [['kernel.perfetto.json','%globaltimer JSON'],['kernel.ncu-rep','NCU report'],['kernel.csv','NCU counters'],['capture.json','采集元数据']]){const a=html('a',label);a.href=s.directory+'/'+(t==='32'?'decode_32':'prefill_4096')+'/'+file;links.append(a)}p.append(links);return p}
function render(){const t=$('shape').value,mode=$('mode').value,b=D.stages[+$('before').value],a=D.stages[+$('after').value],bt=timing(b,t),at=timing(a,t);
 $('cards').replaceChildren(...D.stages.map(s=>{const c=html('div',undefined,'card'),m=timing(s,t);c.append(html('strong',s.label),html('div',m.median_us.toFixed(3)+' µs','value'),html('small',`SGLang ${m.sglang_median_us.toFixed(3)} µs`));return c}));
 const delta=100*(1-at.median_us/bt.median_us);$('delta').textContent=`${bt.median_us.toFixed(3)} → ${at.median_us.toFixed(3)} µs · `+(delta>=0?'降低 ':'增加 ')+Math.abs(delta).toFixed(2)+'%';$('delta').className='metric'+(delta>=0?' positive':'');
 $('context').textContent=`相对同轮 SGLang：${(at.sglang_median_us/at.median_us).toFixed(3)}×。${t==='4096'?'后两项仅修改小 batch decode；prefill 时间差视为波动。':'01 与 00 的 decode 算法和 launch 相同。'}`;
 const svg=el('svg',{viewBox:'0 0 700 160',class:'chart',role:'img','aria-label':'Four-stage graph latency comparison'}),max=Math.max(...D.stages.map(s=>timing(s,t).median_us))*1.13;
 D.stages.forEach((s,i)=>{let m=timing(s,t),y=i*38;svg.append(el('text',{x:0,y:y+22,'font-size':12,fill:'#20394c'},s.short),el('rect',{x:157,y:y+7,width:465*m.median_us/max,height:23,fill:i===3?'#218979':'#4e7d9d',rx:3}),el('text',{x:164+465*m.median_us/max,y:y+23,'font-size':12,fill:'#20394c'},m.median_us.toFixed(3)))});$('bars').replaceChildren(svg);
 const mx=Math.max(b.profiles[t].instrumented_span_us,a.profiles[t].instrumented_span_us)*1.025;$('timelines').replaceChildren(drawTimeline(b,t,mode,mx),drawTimeline(a,t,mode,mx));
 const table=$('phases');table.replaceChildren();const header=html('tr');['阶段','前：interval median µs','后：interval median µs','前：最晚结束 µs','后：最晚结束 µs','前 / 后 intervals'].forEach(x=>header.append(html('th',x)));table.append(header);
 for(const name of ['projection task','Post','Post IO wait','Post A wait','Fn wait','A wait','WGMMA completion gate','Mix dependency wait','Mix partial reduction','Mix affine + Sinkhorn','Mix worker join','Norm dependency wait','Norm compute + store']){const x=b.profiles[t].phases[name],y=a.profiles[t].phases[name];if(!x&&!y)continue;const row=html('tr');[name,x?.median_duration_us.toFixed(3)??'—',y?.median_duration_us.toFixed(3)??'—',x?.last_end_us.toFixed(3)??'—',y?.last_end_us.toFixed(3)??'—',`${x?.count??0} / ${y?.count??0}`].forEach(v=>row.append(html('td',v)));table.append(row)}}
for(const id of ['shape','before','after','mode'])$(id).addEventListener('change',render);render();
</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, out = a.input.expanduser().resolve(), a.output.expanduser().resolve()
    assert root != out and not out.exists(), 'Use a new output directory'
    stages = json.loads((root / 'stages/stages.json').read_text())
    assert [s['key'] for s in stages] == list(STAGES)
    data = dict(timing=json.loads((root / 'staged-comparison.json').read_text()), stages=[])
    out.mkdir(parents=True)
    evidence = out / 'evidence'
    evidence.symlink_to(os.path.relpath(root, out), target_is_directory=True)
    experiments = out / 'experiments'
    experiments.mkdir()
    (experiments / 'decode_trials').symlink_to(os.path.relpath(root/'trials', experiments), target_is_directory=True)
    for i, stage in enumerate(stages):
        s = dict(key=stage['key'], label=stage['label'], short=DIRECTORIES[i][3:],
                 directory=DIRECTORIES[i], source=stage, profiles={})
        for tokens in (4096, 32):
            raw = root / 'profiles' / stage['key'] / str(tokens)
            dest = out / s['directory'] / ('decode_32' if tokens == 32 else 'prefill_4096')
            dest.mkdir(parents=True)
            files = {f'sm90-megamhc-{tokens}.json': 'kernel.perfetto.json', 'kernel.ncu-rep': 'kernel.ncu-rep',
                     'kernel.csv': 'kernel.csv', f'capture-{tokens}.json': 'capture.json', 'ncu-capture.json': 'ncu-capture.json'}
            for name, target in files.items():
                src = raw / name
                assert src.is_file(), src
                (dest / target).symlink_to(os.path.relpath(src, dest))
            s['profiles'][str(tokens)] = summarize_trace(raw / f'sm90-megamhc-{tokens}.json')
        data['stages'].append(s)
    (out / 'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (out / 'index.html').write_text(HTML.replace('__DATA__', payload))
    lines = ['# SM90 MegaMHC 累计 profile 对比', '',
             '打开 [交互索引](index.html)。原始数据保留，阶段目录使用相对 symlink。', '',
             'baseline 指原始 MegaMHC V1；SGLang 为单独的多 kernel 时间参照。', '',
             '| tokens | baseline µs | prefill opt1 µs | decode opt1 µs | decode opt2 µs |',
             '|---:|---:|---:|---:|---:|']
    for c in data['timing']['cases']:
        lines.append('| ' + str(c['tokens']) + ' | ' + ' | '.join(f"{s['median_us']:.4f}" for s in c['stages']) + ' |')
    lines += ['', '各阶段都包含 prefill_4096 / decode_32 的 Perfetto JSON、NCU report、CSV 与采集元数据。', '',
              '打点 launch 与正式 latency 分开。不同 warp 的时间和不是 latency；图中的代表 warp 也可能来自不同 CTA。', '',
              '未采用原型与原因在 [本轮原型计时](experiments/decode_trials/)、[完整实验归档](evidence/) 和仓库实验报告中记录。', '',
              '历史原始目录继续保留：[双 outstanding 与 V1](../sm90-megamhc-double-20261121/README.md)、'
              '[prefill 优化与其他原型](../sm90-megamhc-optimization-20261121/README.md)、'
              '[11-18 V1](../sm90-megamhc-20261118-profiles-v1/README.md)。', '',
              '## 原始归档', '', str(root), '', '各阶段：']
    lines.extend(f"- [{s['label']}]({s['directory']}/)" for s in data['stages'])
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(out), 'html_sha256': hashlib.sha256((out/'index.html').read_bytes()).hexdigest(),
                      'stages': len(stages), 'captures': 8}, ensure_ascii=False))


if __name__ == '__main__':
    main()
