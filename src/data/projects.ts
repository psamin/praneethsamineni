// Every claim here was checked against the repo it describes (code, README,
// git history). Numbers are the ones those repos state. Keep it that way.

export type Project = {
  slug: string;
  title: string;
  /** a few words, shown on the home-page tiles */
  tagline?: string;
  /** one line, shown on cards */
  summary: string;
  /** shown when a card is opened */
  details: string;
  points?: string[];
  when: string;
  /** GitHub URL, or null for a private repo */
  repo: string | null;
  links?: { label: string; href: string }[];
  /** href: make the image a link (e.g. to a demo video) */
  /** video: a muted looping clip that replaces the still (src is its poster) */
  image?: { src: string; alt: string; href?: string; video?: string };
  /** further photos, stacked under the main image */
  moreImages?: { src: string; alt: string }[];
  status?: string;
  featured?: boolean;
  /** extra content rendered inside the card (e.g. the Shakespeare sampler) */
  extra?: "shakespeare";
  /** no "how it works" section: show the summary and bullet points only */
  noDetails?: boolean;
};

const img = (name: string) => `${import.meta.env.BASE_URL}img/${name}`;
const gh = (repo: string) => `https://github.com/psamin/${repo}`;

export const projects: Project[] = [
  {
    slug: "pam",
    featured: true,
    title: "Pam",
    tagline: "AI robot caregiver",
    summary: "A memory-and-fetch assistant for older people: it remembers where you put things, and a robot arm brings them to you.",
    details:
      "A phone camera streams frames into a perception pipeline. YOLOE open-vocabulary detection and BoT-SORT tracking feed a “put-down gate” that fires when an object comes to rest, and Claude then describes where the object landed and logs it to memory. A voice agent (Deepgram STT/TTS over one WebSocket, with Claude tool calls) answers questions like “where are my pills?” It also handles medication doses and reminders, Google Calendar, face recognition and a caregiver dashboard. A 5-motor arm runs an ACT imitation policy: it grabs the bottle, turns, says “Here are your pills,” drops them in your hand and returns home.",
    points: [
      "ACT policy trained on 51 teleoperated episodes",
      "YOLOE + BoT-SORT perception, Claude + Deepgram voice agent",
    ],
    when: "Sep 2026 · HackMIT",
    repo: gh("hackmit"),
    image: { src: img("pam-poster.webp"), video: img("pam.mp4"), alt: "Pam's robot arm on a table next to Praneeth, who is holding a phone" },
  },
  {
    slug: "colony",
    featured: true,
    title: "Colony",
    tagline: "Fleet intelligence for disaster response",
    summary: "A simulated disaster-rescue robot fleet that coordinates entirely through one CockroachDB cluster.",
    details:
      "Scouts, lifters and medics (six robots, nine victims, a magnitude-7.2 aftershock) never message each other. All coordination goes through one CockroachDB cluster: SERIALIZABLE transactions for task claims, leases for failover, changefeeds, and 512-dim vector tables for episodic and semantic memory. A Claude planner on AWS Bedrock runs alongside rule-based plans. A FastAPI server drives a Three.js 3D view. You can kill a robot or a database node live and the mission continues.",
    points: [
      "Rescue rate +31.3 points with coordination (95% CI +23.4 to +39.3, n = 40 paired random disasters)",
      "Victims lost: 2.4 without coordination vs 0.95 with it",
      "Zero tasks lost when a database node is killed",
    ],
    when: "Jul–Aug 2026 · AWS × Cockroach Labs Hackathon",
    repo: gh("project_flock"),
    image: { src: img("colony.webp"), alt: "Colony's 3D digital twin of the disaster site, with robot positions, live fleet stats and the commander console" },
  },
  {
    slug: "instinct",
    featured: true,
    title: "Instinct",
    tagline: "Danger-sensing smart jacket for first responders",
    summary: "A smart jacket for EMTs and soldiers that watches their back while they focus on a patient, and buzzes when something's about to hit them.",
    details:
      "When a medic is doing CPR or assessing a patient, they can't see what's coming from behind. Instinct's cameras track people and objects around them, and a physics-based world model predicts time to collision and miss distance. A task-aware policy then decides whether the risk is worth interrupting them for. It ignores a bystander walking up to watch during CPR, for example, but flags someone closing in fast. Directional haptics in the jacket tell the wearer which way to move. It reasons only about physical motion, never about who looks “dangerous.” I built the phone rig: two iPhones (back and chest) running on-device MediaPipe with their own world models, directional sound and vibration through an Expo app, a CPR rate/pause monitor with a voice coach, and a GPS minimap dashboard.",
    points: [
      "Verified end to end: live camera → tracking → prediction → policy → physical haptic cue",
      "Task-aware: the same approaching person triggers a cue during assessment but not during CPR",
    ],
    when: "Sep 2026 · HackGT",
    repo: null,
    links: [{ label: "Demo video", href: "https://youtu.be/ppUjg5_QX-s" }],
    image: {
      src: "https://img.youtube.com/vi/ppUjg5_QX-s/maxresdefault.jpg",
      alt: "Instinct demo video thumbnail",
      href: "https://youtu.be/ppUjg5_QX-s",
    },
  },
  {
    slug: "robotfpga",
    featured: true,
    title: "Custom-built VLA and GPU",
    tagline: "Controlling our own robotic arm with a GPU on an FPGA",
    status: "In progress",
    summary: "A vision-language-action model for a robot arm, running on our own GPU: a custom int8 accelerator we're building on an AMD Kria FPGA.",
    details:
      "We're basically building our own GPU: a custom int8 accelerator on an AMD Kria FPGA, designed to run a VLA. Then we measure how much task success it keeps, and how much latency and power it saves, compared with an off-the-shelf GPU. I own the software side: a MuJoCo block-sorting scene with an SO-101 arm and language-specified tasks (“put the red cube in the left bin”), a scripted expert for demonstrations, training with behaviour cloning, DART and DAgger, and quantization-aware training that is bit-exact against an int8 reference. Teammates own the FPGA/HLS hardware.",
    points: [
      "Scripted expert: 100/100 on the eval task (95% CI 0.963–1.0)",
      "Target: under 33 ms end-to-end inference on the FPGA",
    ],
    when: "Oct 2026",
    repo: gh("robotfpga"),
  },
  {
    slug: "gpt",
    featured: true,
    title: "GPT from scratch",
    summary: "A character-level GPT in PyTorch, built following Andrej Karpathy's “Let's build GPT” tutorial, trained to write Shakespeare.",
    details:
      "A decoder-only transformer from Attention Is All You Need: multi-head causal self-attention, pre-LayerNorm residual blocks, learned position embeddings and a 4× MLP. It's 6 layers, 6 heads, 384-dim embeddings and a 256-character context, trained on Tiny Shakespeare. The code follows Karpathy's tutorial closely. I trained it for 5,000 steps (about 25 minutes on one NVIDIA L4) to a validation loss of 1.56. Below are unedited samples from that run.",
    when: "Jul 2026",
    repo: gh("transformers"),
    links: [{ label: "Karpathy's tutorial", href: "https://www.youtube.com/watch?v=kCc8FmEb1nY" }],
    extra: "shakespeare",
  },
  {
    slug: "intake",
    title: "AI voice intake platform",
    summary: "A production voice-AI platform that answers and places intake calls for personal-injury law firms.",
    details:
      "Inbound and outbound calls run through Vapi with a custom LLM endpoint that calls Claude. A Temporal worker handles post-call processing (recording, transcript, scored lead, CRM push). An Express server handles webhooks and signed leads from the marketing site. A React console covers leads, cases, live call monitoring with human takeover, campaigns, do-not-call lists, callbacks, retainer e-signature and a client portal. I made the majority of the roughly 1,600 commits.",
    when: "Jun–Jul 2026",
    repo: null,
  },
  {
    slug: "cropai",
    featured: true,
    title: "CropAI",
    summary: "Farm management with a plant-disease CNN built in. 2nd place internationally at TSA Software Development.",
    details:
      "A full-stack farm-management web app: CRUD for crops, finances, soil data, reminders and field plots on Google Maps, with JWT and Google OAuth. Its leaf-analysis endpoint runs a TensorFlow/Keras CNN trained on PlantVillage, and the prediction feeds an AI-generated health report (PDF). There's also an AI chatbot.",
    points: ["2nd place internationally, TSA Software Development", "CNN accuracy 86% after tuning"],
    when: "Jun 2025 · TSA",
    repo: gh("crop2.0"),
  },
  {
    slug: "coding-for-a-change",
    featured: true,
    title: "Coding for a Change",
    summary: "A 501(c)(3) nonprofit that teaches kids to code through tutoring, workshops and hackathons.",
    noDetails: true,
    details:
      "I'm CEO & CTO. I also built its website (Vite, React, TypeScript, shadcn) with Team, Events and Contact pages.",
    points: ["250+ students reached across 6 chapters in 3 states", "2,250+ hours of tutoring, 12+ workshops, 75+ hackathon participants"],
    when: "2025–present",
    repo: gh("code-for-all-ngn"),
    links: [{ label: "codingforachangenpo.org", href: "https://codingforachangenpo.org" }],
    image: { src: img("workshop.webp"), alt: "Praneeth presenting a Coding for a Change workshop in a classroom" },
    moreImages: [{ src: img("coding-for-a-change-zoom.webp"), alt: "A Coding for a Change online class with about twenty students on Zoom" }],
  },
  {
    slug: "robot-policy",
    title: "Tiny robot policy (this site)",
    summary: "The ball-pushing robot on the home page: a 4.7k-parameter MLP trained by imitation, running in your browser.",
    details:
      "A 2D contact simulator written twice, in Python and TypeScript, and checked to agree within 1e-9. A scripted geometric expert generates the demonstrations. A tiny MLP is trained with behaviour cloning plus DART and DAgger, with no RL. The forward pass is hand-written TypeScript, so there's no ML runtime.",
    when: "Oct 2026",
    repo: gh("praneethsamineni"),
  },
  {
    slug: "flashdreams",
    title: "NVIDIA FlashDreams: profiling API",
    status: "PR in progress",
    summary: "Adding NVTX profiling and a stage-timing API to NVIDIA's inference library for interactive video world models.",
    details:
      "I'm adding NVTX ranges, a flashdreams-profile command, an IProfiler interface for per-stage metrics, stage timings stored on the pipeline cache, and frame-rate reporting. That last part includes a fix for an overstated frame rate. The work is on my fork and not yet merged upstream.",
    when: "Sep 2026",
    repo: "https://github.com/psamin/flashdreams/tree/nvtx-profiling",
  },
  {
    slug: "tamil-lens",
    title: "Tamil Lens",
    summary: "Photograph an object and learn its Tamil word: Gemini Vision plus flashcards, quizzes and streaks.",
    details:
      "You photograph an object, Gemini Vision identifies it, and the app returns the English word, the Tamil word and a transliteration. Words go into a personal word bank that feeds flashcards, quizzes, streaks, achievements and stats. It has JWT auth and an admin dashboard. Built with Next.js 15, Flask, SQLAlchemy and Postgres.",
    when: "Sep 2025",
    repo: gh("tamil-lens2.0"),
  },
  {
    slug: "medbill",
    title: "MedBill",
    summary: "Finds overbilling in medical bills by pricing each line item against live Medicare rates.",
    details:
      "MedBill extracts line items from PDF bills with pdfplumber, falling back to Claude Haiku. It compares each charge against Medicare rates from the live CMS data API to calculate overbilling. Cases then move through a role-based workflow between law firm, provider and funder, with negotiated CPT rates and funding batches. Built in two days.",
    when: "Jun 2026",
    repo: gh("medbill"),
    links: [{ label: "Demo video", href: "https://youtu.be/oXoKcJ_Ne78" }],
  },
  {
    slug: "medvoice",
    title: "MedVoice",
    summary: "A Vapi voice intake agent that turns calls into case records, a CRM and prefilled intake forms.",
    details:
      "Prompts live in the repo and are pushed to the assistant with a sync command. An end-of-call webhook saves clients, cases, calls and intake fields to SQLite or Postgres. A CRM layer handles tasks, communications, an audit log and do-not-call opt-outs. After the call, MedVoice emails a prefilled 5-step intake form behind a token link and sends reminders.",
    when: "Jun 2026",
    repo: gh("medvoicevapi"),
  },
  {
    slug: "intake-dashboard",
    title: "Med-legal intake dashboard",
    summary: "Turns raw intake notes or speech into a structured case summary with missing fields and follow-up tasks.",
    details:
      "A Flask backend sends raw intake notes, typed or spoken through the browser's Web Speech API, to Claude with a Pydantic schema. It returns a structured case summary, missing fields, follow-up tasks and a completeness score. Staff review and edit everything in a Next.js dashboard (human in the loop).",
    when: "May 2026",
    repo: gh("waycoReplica"),
    links: [{ label: "Demo video", href: "https://youtu.be/o-id4jpo5ho" }],
  },
  {
    slug: "oss-issue-finder",
    title: "OSS Issue Finder",
    summary: "Ranks open-source issues worth contributing to, from good first PRs to ones that fit long-term goals.",
    details:
      "A resumable GitHub GraphQL backfill into SQLite, with linked-PR reconciliation. Scoring runs in two tiers: a cheap heuristic first, then LLM enrichment through Ollama or Claude with schema validation and a cost log. A Next.js ranked table has a short-term vs long-term timeline slider, filters and a detail drawer.",
    when: "Jul 2026",
    repo: gh("opensourceissues"),
  },
  {
    slug: "repolaunch",
    title: "RepoLaunch",
    summary: "A VS Code extension that scores a repo's launch readiness from 0 to 100 and fixes what's missing.",
    details:
      "RepoLaunch audits the open workspace locally, with no network or telemetry, against 14 weighted checks (README, install docs, license, tests, CI, …). It detects the framework and shows a score card in a sidebar. It generates missing files (CONTRIBUTING, SECURITY, issue templates, CI workflows) behind a diff preview, or hands fixes to Claude Code.",
    when: "Jun 2026",
    repo: gh("RepoLaunch"),
  },
  {
    slug: "recruiter",
    title: "Candidate review board",
    summary: "Type a recruiting prompt, review sourced candidates one card at a time, and send approvals to Google Sheets.",
    details:
      "An Express backend sources candidates through Claude web search and scores them. You review them one card at a time and click Yes, Maybe or No. Approved and Maybe candidates are appended to Google Sheets through a service account. It deliberately doesn't scrape LinkedIn.",
    when: "Jun 2026",
    repo: gh("recruiter"),
  },
  {
    slug: "fisker-it",
    title: "Fisker IT website",
    summary: "Marketing and recruiting site for an SAP consulting firm.",
    details:
      "Home, services, story, recruitment (job listings) and contact pages, with an EmailJS contact form. Deployed on GitHub Pages under the firm's domain.",
    when: "Jun 2025",
    repo: gh("fiskeritinc"),
    links: [{ label: "fiskeritinc.com", href: "https://fiskeritinc.com" }],
  },
  {
    slug: "tasks",
    title: "Tasks",
    summary: "A terminal-style day planner for macOS, built for voice dictation.",
    details:
      "Dictated text is split into tasks on commas, periods and the words “then” and “next”. Times like “7am” or “from 2 to 3pm” are detected and the day is sorted by time. It has vim-style keys and one list per day, stored as local JSON. A Swift wrapper and a LaunchAgent keep it running. Zero dependencies.",
    when: "Mar 2026",
    repo: gh("tasks"),
  },
];

export const featured = projects.filter((p) => p.featured);
