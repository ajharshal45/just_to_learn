export default function Pipeline() {
  const steps = [
    {
      num: 1,
      title: "Fit Scaler",
      desc: "StandardScaler fit on training fold only",
      detail: "transform() applied to test fold — never fit()",
      icon: "📏",
      color: "border-blue-500/40 bg-blue-500/5",
    },
    {
      num: 2,
      title: "Fit Base Estimator",
      desc: "Shallow RandomForest (n=50, max_depth=5)",
      detail: "Trained exclusively on inner training split",
      icon: "🌲",
      color: "border-indigo-500/40 bg-indigo-500/5",
    },
    {
      num: 3,
      title: "Compute SHAP Values",
      desc: "TreeExplainer on training instances",
      detail: "φⱼ(xᵢ) for all i, j — zero test data access",
      icon: "🔮",
      color: "border-violet-500/40 bg-violet-500/5",
    },
    {
      num: 4,
      title: "Compute MASV Weights",
      desc: "Mean Absolute SHAP Values → normalize",
      detail: "w_j = MASV_j / Σ MASV_k",
      icon: "⚖️",
      color: "border-purple-500/40 bg-purple-500/5",
    },
    {
      num: 5,
      title: "Generate Synthetic Samples",
      desc: "SHAP-guided k-NN + interpolation",
      detail: "Minority samples synthesized in training fold only",
      icon: "🧬",
      color: "border-pink-500/40 bg-pink-500/5",
    },
    {
      num: 6,
      title: "Train Final Classifier",
      desc: "LR / Random Forest / XGBoost",
      detail: "Fit on original + synthetic training data",
      icon: "🎯",
      color: "border-rose-500/40 bg-rose-500/5",
    },
    {
      num: 7,
      title: "Evaluate on Test Fold",
      desc: "Untouched held-out test split",
      detail: "PR-AUC, F1, ECE, Kendall's τ computed here",
      icon: "📊",
      color: "border-highlight/40 bg-highlight/5",
    },
  ];

  return (
    <section id="pipeline" className="py-24 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-16">
          <span className="badge bg-highlight/20 text-highlight border border-highlight/30 mb-4">
            Objective 3
          </span>
          <h2 className="section-heading">Zero Data Leakage Protocol</h2>
          <p className="section-subheading">
            Every data-dependent transformation occurs strictly inside the inner training fold.
            The held-out test split is never accessed until final inference.
          </p>
        </div>

        {/* Pipeline steps */}
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-8 top-8 bottom-8 w-px bg-gradient-to-b from-blue-500 via-violet-500 to-highlight hidden lg:block" />

          <div className="space-y-4">
            {steps.map((step, idx) => (
              <div key={step.num} className="flex items-start gap-6 lg:ml-0">
                {/* Number */}
                <div className={`relative flex-shrink-0 w-16 h-16 rounded-2xl border-2 ${step.color} flex items-center justify-center z-10`}>
                  <span className="text-xl">{step.icon}</span>
                  <span className="absolute -top-2 -right-2 w-5 h-5 bg-dark border border-slate-600 rounded-full text-xs flex items-center justify-center text-slate-400 font-bold">
                    {step.num}
                  </span>
                </div>

                {/* Content */}
                <div className={`flex-1 card py-4 px-5 ${step.color} transition-all duration-300`}>
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <h3 className="font-bold text-white">{step.title}</h3>
                      <p className="text-slate-400 text-sm mt-0.5">{step.desc}</p>
                    </div>
                    <div className="formula text-xs shrink-0">{step.detail}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Golden rule */}
        <div className="mt-10 card border-highlight/40 bg-highlight/5 text-center">
          <div className="text-2xl mb-2">🔒</div>
          <h3 className="font-bold text-highlight mb-1">The Golden Rule</h3>
          <p className="text-slate-400 text-sm">
            The test/validation split is <span className="text-white font-semibold">never seen</span> by the scaler, preliminary SHAP estimator, or synthetic oversampler until step 7.
          </p>
        </div>
      </div>
    </section>
  );
}
