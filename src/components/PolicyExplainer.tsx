// "How it works" mini page for the home-page policy. Equations are plain HTML
// (italics, sub/superscripts, fixed-size symbols), so they render the same
// everywhere and nothing stretches or overlaps.
import type { ReactNode } from "react";
import C from "../robot/simConfig.json";
import { policyMeta as meta, thisPolicy } from "../robot/policyInfo";

const Eq = ({ children }: { children: ReactNode }) => <p className="eq">{children}</p>;
const V = ({ children }: { children: ReactNode }) => <i>{children}</i>;

export default function PolicyExplainer({ back }: { back: ReactNode }) {
  const type = thisPolicy.type;
  const pct = (meta.success_rate * 100).toFixed(1);
  return (
    <article className="explainer">
      {back}
      <h1 className="page-title">How the policy works</h1>

      <section>
        <h2>What it does</h2>
        <p>
          Every 33 ms the policy looks at three positions (the robot, the ball and the goal) and outputs the robot's
          velocity. It wins when the ball stays inside the goal.
        </p>
        <Eq>
          <V>a</V> = π(<V>robot</V>, <V>ball</V>, <V>goal</V>)
        </Eq>
        <Eq>
          ‖<V>ball</V> − <V>goal</V>‖ ≤ {C.GOAL_SUCCESS_RADIUS}
        </Eq>
      </section>

      <section>
        <h2>How it learned</h2>
        <p>
          A hand-written controller plays teacher: it walks behind the ball and pushes it toward the goal. The network
          never sees that code. It only learns to copy what the teacher does in each situation:
        </p>
        <Eq>
          <V>L</V>(<V>θ</V>) = 𝔼<sub>
            <V>s</V>
          </sub>{" "}
          ‖ π<sub>
            <V>θ</V>
          </sub>
          (<V>s</V>) − π<sup>*</sup>(<V>s</V>) ‖<sup>2</sup>
        </Eq>
        <p>
          Copying alone drifts: one small mistake leads somewhere the teacher never went. So the network drives, the
          teacher labels what it should have done, and it retrains on those corrections (DAgger). That took it from
          about 30–50% to {pct}% of {meta.eval_episodes.toLocaleString()} unseen random scenes.
        </p>
      </section>

      <section>
        <h2>The network</h2>
        {type === "mlp" && (
          <>
            <p>
              A small multilayer perceptron, {meta.architecture}, with ReLU (σ) between layers:
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
          </>
        )}
        {type === "diffusion" && (
          <>
            <p>{thisPolicy.summary} It is trained to predict the noise ε added to the teacher's next 8 actions:</p>
            <Eq>
              <V>L</V> = 𝔼 ‖ <V>ε</V> − <V>ε</V>
              <sub>
                <V>θ</V>
              </sub>
              (noisy actions, <V>s</V>, <V>k</V>) ‖<sup>2</sup>
            </Eq>
          </>
        )}
        {type === "act" && (
          <>
            <p>{thisPolicy.summary} Each step's action averages every chunk that covers it:</p>
            <Eq>
              <V>a</V>
              <sub>
                <V>t</V>
              </sub>{" "}
              = Σ <V>w</V>
              <sub>
                <V>i</V>
              </sub>{" "}
              <V>A</V>
              <sub>
                <V>i</V>
              </sub>
              [<V>t</V>] / Σ <V>w</V>
              <sub>
                <V>i</V>
              </sub>
            </Eq>
          </>
        )}
        <p>
          {meta.params.toLocaleString()} parameters, {(meta.weight_bytes / 1024).toFixed(1)} KB. It runs on your
          device in about 30 lines of TypeScript: no GPU, no server, no ML library.
        </p>
      </section>
    </article>
  );
}
