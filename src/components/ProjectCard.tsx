import { lazy, Suspense, useId, useState } from "react";
import type { Project } from "../data/projects";

const ShakespeareSampler = lazy(() => import("./ShakespeareSampler"));

/** Large card. Only the summary shows; the rest opens on "how it works". */
export function FeaturedProject({ p }: { p: Project }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <article className={`feature${p.image ? " has-image" : ""}`} id={p.slug}>
      <div className="feature-text">
        <h3 className="row-title">{p.title}</h3>
        <p className="feature-meta">{p.when}</p>
        <p className="row-body">{p.summary}</p>
        <Links p={p} />
        {p.noDetails ? (
          p.points && (
            <ul className="points">
              {p.points.map((x) => <li key={x}>{x}</li>)}
            </ul>
          )
        ) : (
          <>
            <button type="button" className="more-toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
              {open ? "− less" : "+ how it works"}
            </button>
            <div id={id} hidden={!open}>
              <Details p={p} open={open} />
            </div>
          </>
        )}
      </div>
      {p.image && (
        <figure className="feature-image">
          {p.image.href ? (
            <a href={p.image.href} target="_blank" rel="noopener noreferrer" className="play-link">
              <img src={p.image.src} alt={p.image.alt} loading="lazy" decoding="async" />
              <svg className="play-badge" viewBox="0 0 68 48" aria-hidden="true">
                <rect width="68" height="48" rx="12" />
                <path d="M27 15v18l16-9z" />
              </svg>
            </a>
          ) : p.image.video ? (
            <video src={p.image.video} poster={p.image.src} aria-label={p.image.alt} autoPlay muted loop playsInline preload="none" />
          ) : (
            <img src={p.image.src} alt={p.image.alt} loading="lazy" decoding="async" />
          )}
          {p.moreImages?.map((m) => (
            <img key={m.src} src={m.src} alt={m.alt} loading="lazy" decoding="async" />
          ))}
        </figure>
      )}
    </article>
  );
}

/** Grid box. Title + one line; click to open the rest. */
export function ProjectTile({ p }: { p: Project }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <article className={`tile${open ? " is-open" : ""}`} id={p.slug}>
      <button type="button" className="tile-head" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
        <span className="tile-when">{p.when}</span>
        <span className="tile-title">{p.title}</span>
        <span className="tile-summary">{p.summary}</span>
        <span className="tile-plus" aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      <div id={id} className="tile-body" hidden={!open}>
        <Details p={p} open={open} />
      </div>
      <Links p={p} />
    </article>
  );
}

/** Everything behind "how it works": status, write-up, numbers, stack. */
function Details({ p, open }: { p: Project; open: boolean }) {
  return (
    <>
      {p.status && <p className="status-line"><span className="status status-inline">{p.status}</span></p>}
      <p className="details">{p.details}</p>
      {p.points && (
        <ul className="points">
          {p.points.map((x) => <li key={x}>{x}</li>)}
        </ul>
      )}
      {p.extra === "shakespeare" && open && (
        <Suspense fallback={null}>
          <ShakespeareSampler />
        </Suspense>
      )}
      <Tags tags={p.tags} />
    </>
  );
}

function Tags({ tags }: { tags: string[] }) {
  return (
    <ul className="tags">
      {tags.map((t) => <li key={t}>{t}</li>)}
    </ul>
  );
}

function Links({ p }: { p: Project }) {
  return (
    <p className="links">
      {p.repo ? (
        <a href={p.repo} target="_blank" rel="noopener noreferrer">GitHub ↗</a>
      ) : (
        <span className="private">Private repo</span>
      )}
      {p.links?.map((l) => (
        <a key={l.href} href={l.href} target="_blank" rel="noopener noreferrer">{l.label} ↗</a>
      ))}
    </p>
  );
}
