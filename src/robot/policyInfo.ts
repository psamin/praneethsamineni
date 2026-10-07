// Plain-language description of whichever policy was exported. The page
// heading, the paragraph under the demo and the click-the-robot panel all read
// from here, so they always match the shipped weights.
import meta from "./policyMeta.json";

export type PolicyType = "mlp" | "diffusion" | "act";

type Meta = typeof meta & { type?: PolicyType };
export const policyMeta = meta as Meta;
const type: PolicyType = policyMeta.type ?? "mlp";

const NAMES: Record<PolicyType, string> = {
  mlp: "MLP policy",
  diffusion: "Diffusion Policy",
  act: "ACT policy",
};

export const policyName = NAMES[type];

/** What the network is, in one or two sentences. */
export const policySummary: Record<PolicyType, string> = {
  mlp:
    `A deterministic MLP (${policyMeta.architecture}, ReLU) that imitates a scripted expert. ` +
    "It sees only the positions of the robot, the ball and the goal, and 30 times a second it outputs a velocity.",
  diffusion:
    "A scaled-down Diffusion Policy: an MLP denoiser turns random noise into the next 8 actions over 10 DDIM " +
    "steps, conditioned on the positions of the robot, the ball and the goal. The robot executes 4 of them, then plans again.",
  act:
    "A scaled-down ACT (Action Chunking with Transformers): a small transformer predicts the next 8 actions from the " +
    "positions of the robot, the ball and the goal, and overlapping predictions are averaged every step (temporal ensembling).",
};

/** The line under the demo that links to the explainer. */
const SHORT: Record<PolicyType, string> = { mlp: "MLP", diffusion: "diffusion", act: "ACT" };
const CAPTION = {
  lead: "this ",
  highlight: `${SHORT[type]} policy`,
  rest: " allows the robot to dribble the ball to the goal, live in your browser",
};

export const thisPolicy = { type, summary: policySummary[type], caption: CAPTION };
