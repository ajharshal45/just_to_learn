import { leakageData } from "../data/benchmarkData";

export default function LeakageProof() {
  const inflation = leakageData.inflationPercent;
  const leaked = leakageData.leaked;
  const leakFree = leakageData.leakFree;

  return (
    <section id="leakage" className="py-24 px-4 bg-slate-900/30">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-16">
          <span className="badge bg-warning/20 text-warning border border-warning/30 mb-4">
            🚨 Critical Finding
          </span>
          <h2 className="section-heading">The Data Leakage Trap</h2>
          <p className="section-subheading">
            A critical flaw in most published oversampling literature: applying balancing on the
            full dataset before splitting artificially inflates evaluation scores.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-10 items-start">
          {/* Left: proof numbers */}
          <div className="space-y-6">
            {/* Big reveal */}
            <div className="card border-warning/40 bg-warning/5 text-center py-10">
              <div className="text-6xl font-black text-warning mb-2">+{inflation}%</div>
              <div className="text-slate-300 text-lg font-semibold mb-4">Artificial PR-AUC Inflation</div>
              <div className="grid grid-cols-2 gap-4 max-w-xs mx-auto">
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                  <div className="text-red-400 text-xs font-semibold uppercase mb-1">Leaked Pipeline</div>
                  <div className="text-2xl font-black text-red-400">{leaked}</div>
                  <div className="text-xs text-slate-500">PR-AUC</div>
                </div>
                <div className="bg-highlight/10 border border-highlight/30 rounded-xl p-4">
                  <div className="text-highlight text-xs font-semibold uppercase mb-1">Leak-Free</div>
                  <div className="text-2xl font-black text-highlight">{leakFree}</div>
                  <div className="text-xs text-slate-500">PR-AUC</div>
                </div>
              </div>
            </div>

            {/* What causes leakage */}
            <div className="card">
              <h3 className="font-bold text-white mb-4">What Causes Leakage?</h3>
              <div className="space-y-3">
                {[
                  {
                    bad: "❌ Global SMOTE before CV split",
                    good: "✅ SMOTE inside each training fold",
                  },
                  {
                    bad: "❌ Scaler fit on full dataset",
                    good: "✅ Scaler fit on training fold only",
                  },
                  {
                    bad: "❌ SHAP computed on all data",
                    good: "✅ SHAP computed within inner fold",
                  },
                ].map(({ bad, good }) => (
                  <div key={bad} className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-2 text-red-400">{bad}</div>
                    <div className="bg-highlight/5 border border-highlight/20 rounded-lg p-2 text-highlight">{good}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card border-highlight/30">
              <p className="text-sm text-slate-400 leading-relaxed">
                <span className="text-white font-semibold">Conclusion:</span> Publications reporting
                95%+ PR-AUC on imbalanced datasets via global SMOTE are likely contaminated by data
                leakage. Our zero-leakage protocol guarantees{" "}
                <span className="text-highlight">real-world generalizability</span>.
              </p>
            </div>
          </div>

          {/* Right: visual bar comparison */}
          <div className="space-y-6">
            <div className="card">
              <h3 className="font-bold text-white mb-6">PR-AUC Comparison on Kaggle Stroke Dataset</h3>

              {/* Visual bars */}
              <div className="space-y-5">
                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-2">
                    <span className="text-red-400 font-semibold">Leaked Pipeline (Global SMOTE)</span>
                    <span className="text-red-400 font-bold">{leaked}</span>
                  </div>
                  <div className="h-8 bg-slate-900 rounded-lg overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-lg flex items-center justify-end pr-3 text-xs font-bold text-white transition-all duration-1000"
                      style={{ width: `${(leaked * 100)}%` }}
                    >
                      FAKE
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-2">
                    <span className="text-highlight font-semibold">Leak-Free Pipeline (Ours)</span>
                    <span className="text-highlight font-bold">{leakFree}</span>
                  </div>
                  <div className="h-8 bg-slate-900 rounded-lg overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-highlight to-emerald-400 rounded-lg flex items-center justify-end pr-3 text-xs font-bold text-white"
                      style={{ width: `${(leakFree * 100)}%` }}
                    >
                      REAL
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 text-center text-sm text-warning font-semibold">
                Delta: +{leakageData.inflationDelta} PR-AUC artificial inflation
              </div>
            </div>

            {/* Figure 2 */}
            <div className="card p-3">
              <img
                src="/assets/proofs/fig2_data_leakage_trap_proof.png"
                alt="Data Leakage Trap Proof - Figure 2"
                className="w-full rounded-xl"
                loading="lazy"
              />
              <p className="text-xs text-slate-500 text-center mt-2">
                Fig 2: Empirical demonstration of data leakage trap
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
