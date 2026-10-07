// Real samples from my training run of the char-level GPT, generated offline
// (the 10.8M-param model is too big to ship for a portfolio page).
// Typed out character by character, then the next sample starts on its own.
// Typing pauses while the panel is off-screen.
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
  const [visible, setVisible] = useState(true);
  const box = useRef<HTMLPreElement>(null);
  const text = run.samples[i].trim();
  const reduced = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    if (box.current) io.observe(box.current);
    return () => io.disconnect();
  }, []);

  useEffect(() => setN(reduced ? text.length : 0), [text, reduced]);

  useEffect(() => {
    if (reduced || !visible) return;
    if (n >= text.length) {
      const id = setTimeout(() => setI((k) => (k + 1) % run.samples.length), 2500); // next sample
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => setN((k) => Math.min(text.length, k + 2)), 22);
    return () => clearTimeout(id);
  }, [n, text, visible, reduced]);

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
          next ↻
        </button>
      </div>
      <pre ref={box} className="sampler-text" aria-live="off">
        {text.slice(0, n)}
        {n < text.length && <span className="cursor" aria-hidden="true" />}
      </pre>
    </div>
  );
}
