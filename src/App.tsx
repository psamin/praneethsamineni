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
  // brand / organisation link in the bio (opens in a new tab)
  const L = ({ to, children }: { to: string; children: string }) => (
    <a href={to} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
  return (
    <>
      <section className="intro">
        <img className="portrait" src={img("praneeth.webp")} alt="Praneeth Samineni" width={136} height={136} />
        <div>
          <p className="intro-hey">hey! my name is</p>
          <h1 className="intro-name">Praneeth Samineni</h1>
          <SocialIcons />
        </div>
      </section>

      <section className="bio">
        <p>
          I'm a Computer Science and Mathematics double major at <L to="https://www.gatech.edu/">Georgia Tech</L>.
        </p>
        <p>
          Currently, I'm contributing to <L to="https://github.com/NVIDIA/flashdreams">NVIDIA FlashDreams</L> to help solve
          inference bottlenecks in world models, researching the mechanistic interpretability of World Action Models at{" "}
          <L to="https://www.eleuther.ai/">EleutherAI</L>, working on LLM inference research at{" "}
          <L to="https://sail.cc.gatech.edu/">Georgia Tech SAIL</L>, and building a humanoid robot from my dorm room.
        </p>
        <p>
          My interests lie in combining high-level agentic control with low-level robot policies to quickly deploy
          robots. I also post about what I'm learning every day on{" "}
          <L to="https://x.com/praneethsamine">X</L>!
        </p>
        <p>Previously, I:</p>
        <ul>
          <li>
            Worked as a Forward Deployed Engineer at <L to="https://wayco.ai/">Wayco AI</L> (YC W26), building voice AI
            agents.
          </li>
          <li>
            Founded a 501(c)(3) <L to="https://codingforachangenpo.org/">nonprofit</L> to teach Python and AI to 300+
            middle schoolers across 3 states, and hosted a hackathon with 75+ participants.
          </li>
          <li>
            Won multiple hackathons and software development competitions hosted by{" "}
            <L to="https://www.ycombinator.com/">YC</L>, <L to="https://tsaweb.org/">TSA</L>, and{" "}
            <L to="https://www.fbla.org/">FBLA</L>.
          </li>
        </ul>
      </section>

      <section id="demo" className="sec sec-plain demo-below">
        <div className="sec-body">
          <p className="demo-title">This is a robot policy, live in your browser!</p>
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
    </>
  );
}


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
