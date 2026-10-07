// "How it works" page for the home-page policy. Equations are plain HTML
// (italics, sub/superscripts, fixed-size symbols): no math library, and
// nothing stretches or overlaps.
import type { ReactNode } from "react";
import C from "../robot/simConfig.json";
import { policyMeta as meta, thisPolicy } from "../robot/policyInfo";

// Behavior-cloning-only success of the shipped model (abs_32_bc in results/evaluation.json).
const BC_ONLY_SUCCESS = 31.4;

const Eq = ({ children }: { children: ReactNode }) => <p className="eq">{children}</p>;
const V = ({ children }: { children: ReactNode }) => <i>{children}</i>;
const Term = ({ children }: { children: ReactNode }) => <b className="term">{children}</b>;
const Pi = ({ star }: { star?: boolean }) =>
  star ? (
    <>
      π<sup>*</sup>
    </>
  ) : (
    <>
      π<sub>
        <V>θ</V>
      </sub>
    </>
  );
const State = () => (
  <>
    (<V>x</V>
    <sub>r</sub>, <V>y</V>
    <sub>r</sub>, <V>x</V>
    <sub>b</sub>, <V>y</V>
    <sub>b</sub>, <V>x</V>
    <sub>g</sub>, <V>y</V>
    <sub>g</sub>)
  </>
);
const Vel = () => (
  <>
    (<V>v</V>
    <sub>x</sub>, <V>v</V>
    <sub>y</sub>)
  </>
);

export default function PolicyExplainer({ back }: { back: ReactNode }) {
  const isMlp = thisPolicy.type === "mlp";
  return (
    <article className="explainer">
      {back}
      <h1 className="page-title">How it works</h1>

      <section>
        <h2>How the policy works</h2>
        <p>
          Every {Math.round(C.PHYSICS_DT * C.SUBSTEPS_PER_ACTION * 1000)} ms, the MLP policy reads the positions of the
          robot, ball, and goal and outputs the robot's velocity.
        </p>
        <Eq>
          <V>s</V> = <State />
        </Eq>
        <Eq>
          <V>a</V> = <Pi />(<V>s</V>) = <Vel />
        </Eq>
        <p>
          A rollout succeeds when the ball stays within {C.GOAL_SUCCESS_RADIUS} units of the goal for{" "}
          {C.SUCCESS_HOLD_STEPS} consecutive steps:
        </p>
        <Eq>
          ‖<V>ball</V> − <V>goal</V>‖ ≤ {C.GOAL_SUCCESS_RADIUS}
        </Eq>
      </section>

      <section>
        <h2>How it learned</h2>
        <p>
          A hand-coded controller serves as the <Term>expert policy</Term>. The MLP policy is first trained with{" "}
          <Term>behavior cloning</Term> to match the expert's action at each state:
        </p>
        <Eq>
          <Pi />(<V>s</V>) ≈ <Pi star />(<V>s</V>)
        </Eq>
        <p>Training then minimizes the error between the policy's action and the expert action:</p>
        <Eq>
          <V>L</V>(<V>θ</V>) = 𝔼<sub>
            <V>s</V>
          </sub>
          [ ‖ <Pi />(<V>s</V>) − <Pi star />(<V>s</V>) ‖<sup>2</sup> ]
        </Eq>
        <p>
          However, behavior cloning only trains on states from the original demonstrations. If the learned policy
          makes a small mistake, it can reach states it has never seen before.
        </p>
        <p>
          To correct for this, <Term>DAgger</Term> lets the learned policy roll out, labels the states it visits with
          the expert policy, and retrains on the expanded dataset:
        </p>
        <Eq>
          rollout <Pi /> → label with <Pi star /> → retrain
        </Eq>
        <p>
          After this, success increased from <b>{BC_ONLY_SUCCESS}% to {(meta.success_rate * 100).toFixed(1)}%</b>{" "}
          across <b>{meta.eval_episodes.toLocaleString()} unseen random scenes</b>.
        </p>
      </section>

      <section>
        <h2>MLP policy</h2>
        {isMlp ? (
          <>
            <p>The learned policy is a small multilayer perceptron:</p>
            <Eq>{meta.architecture}</Eq>
            <p>
              The six inputs are <State />, the x–y positions of the robot, ball, and goal, and the two outputs are{" "}
              <Vel />, the velocity the robot should move at. The forward pass is:
            </p>
            <Eq>
              <V>a</V> = <V>W</V>
              <sub>3</sub> σ(<V>W</V>
              <sub>2</sub> σ(<V>W</V>
              <sub>1</sub>
              <V>s</V> + <V>b</V>
              <sub>1</sub>) + <V>b</V>
              <sub>2</sub>) + <V>b</V>
              <sub>3</sub>
            </Eq>
            <p>with ReLU activations:</p>
            <Eq>
              σ(<V>x</V>) = max(0, <V>x</V>)
            </Eq>
          </>
        ) : (
          <p>{thisPolicy.summary}</p>
        )}
        <p>
          The policy has <b>{meta.params.toLocaleString()} parameters</b>, occupies about{" "}
          <b>{(meta.weight_bytes / 1024).toFixed(1)} KB</b>, and runs entirely in the browser with TypeScript. No GPU,
          server, or ML framework is required.
        </p>
      </section>
    </article>
  );
}
