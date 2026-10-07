// Real samples from my training run of the char-level GPT, generated offline
// (the 10.8M-param model is too big to ship for a portfolio page).
// Typed out character by character to show how it generates.
import { useEffect, useRef, useState } from "react";
import gpt from "../data/shakespeare.json";

type Run = { params_M: number; val_loss: number | null; samples: string[] };
const run = gpt as Run;

export default function ShakespeareSampler() {
  if (!run.samples.length) {
    return <p className="sampler sampler-empty">Training run in progress. Samples will appear here when it finishes.</p>;
  }
  return <Sampler />;
}

function Sampler() {
  const [i, setI] = useState(0);
  const [n, setN] = useState(0);
  const box = useRef<HTMLPreElement>(null);
  const text = run.samples[i].trim();

  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { setN(text.length); return; }
    setN(0);
    const id = setInterval(() => setN((k) => (k >= text.length ? k : k + 3)), 16);
    return () => clearInterval(id);
  }, [text]);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [n]);

  return (
    <div className="sampler">
      <div className="sampler-head">
        <span>
          sample {i + 1}/{run.samples.length} · {run.params_M.toFixed(1)}M params · val loss {run.val_loss?.toFixed(2)}
        </span>
        <button type="button" onClick={() => setI((i + 1) % run.samples.length)}>
          ↻ another
        </button>
      </div>
      <pre ref={box} className="sampler-text" aria-live="off">
        {text.slice(0, n)}
        {n < text.length && <span className="cursor" aria-hidden="true" />}
      </pre>
    </div>
  );
}
