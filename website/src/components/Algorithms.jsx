export default function Algorithms() {
  const algorithms = [
    {
      id: "filtered",
      title: "SHAP-Filtered SMOTE",
      subtitle: "Pre-Filtering Subspace Variant",
      badge: "Method 1",
      badgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30",
      icon: "🔍",
      description:
        "Prunes uninformative noise dimensions using Mean Absolute SHAP Values (MASV) computed on the inner training set. Nearest-neighbour graph construction happens in the high-signal feature subspace only.",
      steps: [
        "Fit base estimator (RF) on training fold",
        "Compute SHAP values → MASV per feature",
        "Select top features by cumulative MASV ≥ 90%",
        "Build k-NN graph in reduced subspace",
        "Interpolate synthetic samples in full feature space",
      ],
      formula: "w_j = MASV_j / Σₖ MASV_k",
      advantage: "Prevents noise features from corrupting neighbor selection",
      color: "border-indigo-500/30 hover:border-indigo-500/60",
      iconBg: "bg-indigo-500/10",
    },
    {
      id: "weighted",
      title: "SHAP-Weighted MASV-SMOTE",
      subtitle: "In-Process Geometry Modification",
      badge: "Method 2",
      badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
      icon: "⚖️",
      description:
        "Retains all features but modulates the k-nearest-neighbour distance metric using normalized MASV weights. Features with high predictive attribution dominate neighborhood selection.",
      steps: [
        "Fit base estimator (RF) on training fold",
        "Compute SHAP values → normalized weights w_j",
        "Apply x_weighted = x × √(w) transformation",
        "Build k-NN in weighted metric space",
        "Interpolate in original feature space",
      ],
      formula: "d_W(xₐ, x_b) = √( Σⱼ wⱼ·(xₐⱼ − x_bⱼ)² )",
      advantage: "Game-theoretic weighting accounts for non-linear feature interactions",
      color: "border-emerald-500/30 hover:border-emerald-500/60",
      iconBg: "bg-emerald-500/10",
    },
  ];

  return (
    <section id="algorithms" className="py-24 px-4 bg-slate-900/30">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-16">
          <span className="badge bg-primary/20 text-primary border border-primary/30 mb-4">
            Objective 2
          </span>
          <h2 className="section-heading">Our Algorithms</h2>
          <p className="section-subheading">
            Two complementary strategies for incorporating SHAP feature attributions into synthetic
            oversampling — both strictly leak-free.
          </p>
        </div>

        {/* MASV shared explanation */}
        <div className="card mb-10 border-primary/30">
          <h3 className="text-base font-bold text-primary mb-3">
            Shared Foundation: Mean Absolute SHAP Value (MASV) Weighting
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3 text-sm text-slate-400">
              <p>For each feature <span className="text-white font-mono">j</span>, MASV aggregates individual Shapley attributions across all training instances:</p>
              <div className="formula">MASV_j = (1/N) Σᵢ |φⱼ(xᵢ)|</div>
              <p>Normalized weights form a probability simplex (sum to 1):</p>
              <div className="formula">w_j = MASV_j / Σₖ MASV_k   →   Σⱼ wⱼ = 1, wⱼ ≥ 0</div>
            </div>
            <div className="text-sm text-slate-400 space-y-2">
              <p className="font-semibold text-white">Why MASV over Gini / Mutual Information?</p>
              <div className="space-y-2">
                {[
                  ["❌ Gini Impurity", "Biased toward high-cardinality continuous features"],
                  ["❌ Mutual Information", "Univariate — ignores feature interaction effects"],
                  ["✅ TreeSHAP MASV", "Game-theoretic, captures non-linear conditional interactions"],
                ].map(([method, desc]) => (
                  <div key={method} className="flex gap-3 items-start">
                    <span className="font-mono text-xs shrink-0 mt-0.5 w-36">{method}</span>
                    <span className="text-xs text-slate-500">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Algorithm cards */}
        <div className="grid lg:grid-cols-2 gap-8">
          {algorithms.map((algo) => (
            <div key={algo.id} className={`card transition-colors duration-300 ${algo.color}`}>
              <div className="flex items-start justify-between mb-4">
                <div className={`w-12 h-12 rounded-xl ${algo.iconBg} flex items-center justify-center text-2xl`}>
                  {algo.icon}
                </div>
                <span className={`badge border ${algo.badgeColor}`}>{algo.badge}</span>
              </div>

              <h3 className="text-xl font-bold text-white mb-1">{algo.title}</h3>
              <p className="text-sm text-slate-500 mb-4">{algo.subtitle}</p>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">{algo.description}</p>

              {/* Steps */}
              <div className="mb-6">
                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Algorithm Steps</h4>
                <ol className="space-y-2">
                  {algo.steps.map((step, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-primary/20 text-primary text-xs flex items-center justify-center font-bold mt-0.5">
                        {i + 1}
                      </span>
                      <span className="text-slate-400">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>

              {/* Formula */}
              <div className="mb-4">
                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Key Formula</h4>
                <div className="formula text-xs overflow-x-auto">{algo.formula}</div>
              </div>

              {/* Advantage */}
              <div className="flex items-start gap-2 text-xs text-highlight bg-highlight/5 border border-highlight/20 rounded-lg px-3 py-2">
                <span className="mt-0.5">✓</span>
                <span>{algo.advantage}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
