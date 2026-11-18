export default function Problem() {
  const limitations = [
    {
      icon: "⚖️",
      title: "Feature Agnosticism",
      description:
        "Euclidean distance weights all dimensions equally. Noise features and uninformative variables influence neighbor selection as heavily as key predictive drivers.",
      formula: "d(xₐ, x_b) = √Σⱼ (xₐⱼ − x_bⱼ)²",
    },
    {
      icon: "🚧",
      title: "Boundary Distortion",
      description:
        "Interpolating uniformly across multi-dimensional feature space risks generating synthetic instances inside majority-class territories, creating artificial class overlap and label noise.",
      formula: "x_new = xᵢ + λ·(x_zᵢ − xᵢ),  λ ~ U(0,1)",
    },
    {
      icon: "📐",
      title: "Curse of Dimensionality",
      description:
        "As the number of noise features p₂ grows, isotropic Euclidean neighborhoods collapse and fail to reflect the true manifold of the minority distribution.",
      formula: "lim(p₂→∞) Var(d)/Mean(d) → 0",
    },
  ];

  return (
    <section id="problem" className="py-24 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="mb-16">
          <span className="badge bg-warning/20 text-warning border border-warning/30 mb-4">
            The Problem
          </span>
          <h2 className="section-heading">Why Standard SMOTE Falls Short</h2>
          <p className="section-subheading">
            SMOTE's isotropic Euclidean distance treats all features uniformly, corrupting synthetic
            minority samples with noise and creating false class boundaries.
          </p>
        </div>

        {/* Research gap pipeline */}
        <div className="card mb-12 glow">
          <h3 className="text-lg font-bold text-white mb-6">The Paradigm Shift</h3>
          <div className="space-y-4">
            {/* Old paradigm */}
            <div className="bg-slate-900/60 rounded-xl p-4 border border-red-500/20">
              <div className="text-xs text-red-400 font-semibold mb-2 uppercase tracking-wider">❌ Conventional Paradigm</div>
              <div className="flex flex-wrap items-center gap-2 text-sm font-mono text-slate-300">
                {["Imbalanced Data", "Unweighted SMOTE", "Train Model", "SHAP (Post-hoc)"].map((step, i, arr) => (
                  <span key={step} className="flex items-center gap-2">
                    <span className="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg">{step}</span>
                    {i < arr.length - 1 && <span className="text-slate-600">→</span>}
                  </span>
                ))}
              </div>
              <p className="text-xs text-red-400/70 mt-2">SHAP used only as after-the-fact diagnostic — does NOT improve data quality</p>
            </div>

            {/* New paradigm */}
            <div className="bg-slate-900/60 rounded-xl p-4 border border-highlight/20">
              <div className="text-xs text-highlight font-semibold mb-2 uppercase tracking-wider">✅ Our Proposed Paradigm</div>
              <div className="flex flex-wrap items-center gap-2 text-sm font-mono text-slate-300">
                {["Imbalanced Data", "Base Estimator", "SHAP MASV Weights", "SHAP-Guided SMOTE", "Classifier", "Evaluate"].map((step, i, arr) => (
                  <span key={step} className="flex items-center gap-2">
                    <span className={`px-3 py-1.5 rounded-lg border ${
                      step.includes("SHAP") ? "bg-primary/20 border-primary/50 text-primary" : "bg-slate-800 border-slate-700"
                    }`}>{step}</span>
                    {i < arr.length - 1 && <span className="text-highlight">→</span>}
                  </span>
                ))}
              </div>
              <p className="text-xs text-highlight/70 mt-2">SHAP moved upstream — guides synthetic sample placement BEFORE model training</p>
            </div>
          </div>
        </div>

        {/* Limitation cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {limitations.map((lim) => (
            <div key={lim.title} className="card group hover:border-warning/40">
              <div className="text-3xl mb-4">{lim.icon}</div>
              <h3 className="text-lg font-bold text-white mb-3">{lim.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">{lim.description}</p>
              <div className="formula text-xs overflow-x-auto">{lim.formula}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
