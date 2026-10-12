import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const initial = {
  depth:{baseline:42, proposed:27},
  swaps:{baseline:18, proposed:7},
  gates:{baseline:103, proposed:76},
  time:{baseline:0.42, proposed:0.81},
  fidelity:{baseline:0.81, proposed:0.91},
  state:{baseline:0.98, proposed:0.99}
};

const labels = [
  ['depth','Circuit depth',''],
  ['swaps','SWAP gates',''],
  ['gates','Total gates',''],
  ['time','Compilation time','s'],
  ['fidelity','Estimated fidelity',''],
  ['state','State fidelity','']
];

function App(){
  const [circuit,setCircuit]=useState('QFT-5');
  const [hardware,setHardware]=useState('8-Qubit Grid');
  const [results,setResults]=useState(initial);
  const [compiled,setCompiled]=useState(false);

  const reduction=(b,p)=>((b-p)/b*100).toFixed(1);
  const compile=()=>{setCompiled(true);};

  return <div className="page">
    <header className="topbar">
      <div className="brand"><span className="orb">⚛</span><div><div className="title">QUANTUM COMPILER</div><div className="subtitle">Qubit Allocation &amp; Depth Minimisation</div></div></div>
      <div className="status"><span></span> SYSTEM READY</div>
    </header>

    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">COMPILATION ANALYSIS</p>
          <h1>Compare your <em>baseline</em> and proposed compiler.</h1>
          <p className="muted">A clean comparison view for SABRE/QMAP baselines versus the GNN + RL approach.</p>
        </div>
        <div className="controls">
          <label>CIRCUIT<select value={circuit} onChange={e=>setCircuit(e.target.value)}><option>QFT-5</option><option>GHZ-5</option><option>Grover-5</option><option>QFT-8</option></select></label>
          <label>HARDWARE<select value={hardware} onChange={e=>setHardware(e.target.value)}><option>8-Qubit Grid</option><option>8-Qubit Linear</option></select></label>
          <button onClick={compile}>{compiled ? 'COMPILED ✓' : 'COMPILE CIRCUIT'}</button>
        </div>
      </section>

      <section className="comparison card">
        <div className="cardhead"><div><p className="eyebrow">RESULTS</p><h2>Baseline <span>vs</span> Proposed</h2></div><div className="badge">{circuit} · {hardware}</div></div>
        <div className="table">
          <div className="row head"><div></div><div>Baseline</div><div>Proposed</div><div>Improvement</div></div>
          {labels.map(([key,label,suffix])=><div className="row" key={key}>
            <div className="metric">{label}</div>
            <div>{results[key].baseline}{suffix}</div>
            <div className="proposed">{results[key].proposed}{suffix}</div>
            <div className={(key==='time')?'neutral':'good'}>{key==='time'?`+${((results[key].proposed-results[key].baseline)/results[key].baseline*100).toFixed(1)}%`:`↓ ${reduction(results[key].baseline,results[key].proposed)}%`}</div>
          </div>)}
        </div>
      </section>

      <section className="grid2">
        <div className="card compact"><div className="cardhead"><div><p className="eyebrow">DEPTH</p><h2>Circuit depth</h2></div><strong>{reduction(results.depth.baseline,results.depth.proposed)}% <small>reduction</small></strong></div><div className="bars"><div><span>Baseline</span><i style={{width:'100%'}}></i><b>{results.depth.baseline}</b></div><div><span>Proposed</span><i style={{width:`${results.depth.proposed/results.depth.baseline*100}%`}}></i><b>{results.depth.proposed}</b></div></div></div>
        <div className="card compact"><div className="cardhead"><div><p className="eyebrow">SWAP OVERHEAD</p><h2>SWAP gates</h2></div><strong>{reduction(results.swaps.baseline,results.swaps.proposed)}% <small>reduction</small></strong></div><div className="bars"><div><span>Baseline</span><i style={{width:'100%'}}></i><b>{results.swaps.baseline}</b></div><div><span>Proposed</span><i style={{width:`${results.swaps.proposed/results.swaps.baseline*100}%`}}></i><b>{results.swaps.proposed}</b></div></div></div>
      </section>

      <section className="circuit card">
        <div className="cardhead"><div><p className="eyebrow">VISUALISATION</p><h2>Compilation snapshot</h2></div><div className="legend"><span className="dot"></span> logical <span className="dot physical"></span> physical</div></div>
        <div className="circuitbox">
          <div className="circuitlabel">BASELINE</div><div className="wires"><span>q0 ─── H ─── ● ───────── X ───</span><span>q1 ───────── X ─── ● ───────</span><span>q2 ─────────────── X ────────</span></div>
          <div className="divider"></div>
          <div className="circuitlabel proposedlabel">PROPOSED · GNN + RL</div><div className="wires"><span>q0 ─── H ─── ● ──────────────</span><span>q1 ───── SWAP ─ X ─── ● ─────</span><span>q2 ────────────────── X ─────</span></div>
        </div>
      </section>
    </main>
    <footer>QUANTUM COMPILER · {new Date().getFullYear()} · RESULTS DASHBOARD</footer>
  </div>
}
createRoot(document.getElementById('root')).render(<App/>);
