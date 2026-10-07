import { useEffect, useState } from "react";
import { FeaturedProject, ProjectTile } from "./components/ProjectCard";
import { featured, projects } from "./data/projects";
import PolicyExplainer from "./components/PolicyExplainer";
import SocialIcons from "./components/SocialIcons";
import { href, navigate, usePath } from "./router";
import BallPushDemo from "./robot/BallPushDemo";
import { thisPolicy } from "./robot/policyInfo";

const img = (name: string) => `${import.meta.env.BASE_URL}img/${name}`;

export default function App() {
  const path = usePath();
  const page = PAGES.find((p) => p.path === path) ?? PAGES[0];
  const [menuOpen, setMenuOpen] = useState(false);

  // close the phone menu on navigation and on Escape
  useEffect(() => setMenuOpen(false), [path]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMenuOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    document.title = page.path === "/" ? "Praneeth Samineni" : `${page.label} · Praneeth Samineni`;
    // after a page switch, land on the #section if there is one
    const target = window.location.hash && document.querySelector(window.location.hash);
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [path, page]);

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>

      <header className="nav">
        <div className="wrap nav-inner">
          <a className="brand" href={href("/")} onClick={navigate}>Praneeth Samineni</a>
          <nav className="nav-links" aria-label="Primary">
            <NavLinks page={page} />
          </nav>
          <button
            type="button"
            className="menu-button"
            aria-label="Open menu"
            aria-expanded={menuOpen}
            aria-controls="side-menu"
            onClick={() => setMenuOpen(true)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>
      </header>

      {/* phone-only sidebar */}
      <div className={`side-backdrop${menuOpen ? " is-open" : ""}`} onClick={() => setMenuOpen(false)} aria-hidden="true" />
      <nav id="side-menu" className={`side-menu${menuOpen ? " is-open" : ""}`} aria-label="Menu" aria-hidden={!menuOpen}>
        <button type="button" className="side-close" aria-label="Close menu" onClick={() => setMenuOpen(false)}>
          &times;
        </button>
        <NavLinks page={page} onPick={() => setMenuOpen(false)} />
      </nav>

      <main id="main">
        <div className="wrap" id="top">
          <page.Component />
          <blockquote className="quote">
            <p>If you believe in robotics, <br />robotics will believe in you.&rdquo;</p>
            <cite>Jim Fan</cite>
          </blockquote>
          <blockquote className="quote">
            <p>If you believe in deep learning, <br />deep learning will believe in you.&rdquo;</p>
            <cite>Ilya Sutskever</cite>
          </blockquote>
        </div>
      </main>

      <footer>
        <div className="wrap footer-inner">
          <span>&copy; {new Date().getFullYear()} Praneeth Samineni</span>
        </div>
      </footer>
    </>
  );
}

function HomePage() {
  return (
    <>
      <section className="intro">
        <img className="portrait" src={img("praneeth.webp")} alt="Praneeth Samineni" width={136} height={136} />
        <div>
          <p className="intro-hey">hey! my name is</p>
          <h1 className="intro-name">Praneeth Samineni</h1>
          <p className="intro-line">
            I'm currently double majoring in <strong>CS + Math</strong> at <strong>Georgia Tech</strong>, with a minor in{" "}
            <strong>Robotics</strong>. I like messing with robotics and deep learning.
          </p>
          <SocialIcons />
        </div>
      </section>

      <section id="demo" className="sec sec-plain">
        <div className="sec-body">
          <BallPushDemo />
          <p className="demo-link">
            <a href={href("/policy")} onClick={navigate}>
              {thisPolicy.caption.lead}
              <span className="demo-policy">{thisPolicy.caption.highlight}</span>
              {thisPolicy.caption.rest} · see how it works <span className="arrow" aria-hidden="true">→</span>
            </a>
          </p>
        </div>
      </section>

      <section className="sec">
        <h2>Projects</h2>
        <ul className="shelf sec-body">
          {HOME_PROJECTS.map((slug) => {
            const p = projects.find((x) => x.slug === slug)!;
            return (
              <li key={slug}>
                <a className="shelf-item" href={href(`/projects#${slug}`)} onClick={navigate}>
                  {p.image ? (
                    <img src={p.image.src} alt="" loading="lazy" decoding="async" />
                  ) : (
                    <span className="shelf-soon">Coming soon</span>
                  )}
                  <span className="shelf-title">{p.title}</span>
                  <span className="shelf-desc">{p.tagline ?? p.summary}</span>
                </a>
              </li>
            );
          })}
          <li>
            <a className="shelf-item shelf-all" href={href("/projects")} onClick={navigate}>
              <span className="shelf-title">
                All {projects.length} projects <span className="arrow" aria-hidden="true">→</span>
              </span>
            </a>
          </li>
        </ul>
      </section>
    </>
  );
}

/** Projects linked from the home page, in order. */
const HOME_PROJECTS = ["pam", "colony", "instinct", "robotfpga"];

function ProjectsPage() {
  const rest = projects.filter((p) => !p.featured);
  return (
    <>
      <section className="projects-page">
        <h1 className="page-title">Projects</h1>
        <div className="features">
          {featured.map((p) => (
            <FeaturedProject key={p.slug} p={p} />
          ))}
        </div>
      </section>

      <section id="more" className="sec">
        <h2>More projects</h2>
        <div className="tile-grid sec-body">
          {rest.map((p) => (
            <ProjectTile key={p.slug} p={p} />
          ))}
        </div>
      </section>
    </>
  );
}

function BlogPage() {
  return (
    <section className="soon">
      <p className="soon-word">
        Coming soon<span className="cursor" aria-hidden="true" />
      </p>
    </section>
  );
}

function PolicyPage() {
  return (
    <PolicyExplainer
      back={
        <a className="back-link" href={href("/")} onClick={navigate}>← Back</a>
      }
    />
  );
}

const PAGES = [
  { path: "/", label: "Me", Component: HomePage, nav: true },
  { path: "/projects", label: "Projects", Component: ProjectsPage, nav: true },
  { path: "/blog", label: "Blog", Component: BlogPage, nav: true },
  { path: "/policy", label: "How the policy works", Component: PolicyPage, nav: false },
];

function NavLinks({ page, onPick }: { page: (typeof PAGES)[number]; onPick?: () => void }) {
  return (
    <>
      {PAGES.filter((p) => p.nav).map((p) => (
        <a
          key={p.path}
          href={href(p.path)}
          onClick={(e) => { navigate(e); onPick?.(); }}
          aria-current={p === page ? "page" : undefined}
        >
          {p.label}
        </a>
      ))}
    </>
  );
}
