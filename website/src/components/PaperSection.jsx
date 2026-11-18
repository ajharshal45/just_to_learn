import { useState } from "react";

const bibtex = `@article{shapsmote2024,
  title   = {Explainability-Aware Synthetic Oversampling for
             Imbalanced Tabular Classification},
  author  = {Hasan, Rachit and others},
  journal = {arXiv preprint},
  year    = {2024},
  note    = {A Systematic Literature Review and Empirical
             Benchmark of SHAP-Guided SMOTE Methods}
}`;

const abstract = `Imbalanced tabular data represents a persistent challenge in applied machine learning, 
particularly in high-stakes domains such as fraud detection, medical diagnostics, fault prediction, 
and network security. Conventional synthetic oversampling methods—most notably SMOTE—generate 
synthetic minority instances by interpolating along Euclidean line segments between nearest neighbours. 
However, standard Euclidean distance treats all features uniformly, ignoring their true predictive 
contribution and allowing uninformative or noisy features to distort synthetic sample placement.

This work shifts the paradigm: rather than applying SHAP as an after-the-fact explanation tool, 
we utilize SHAP-derived Mean Absolute SHAP Values (MASV) upstream during data pre-processing to 
guide synthetic sample generation. We formulate two algorithms—SHAP-Filtered SMOTE (subspace 
pre-filtering) and SHAP-Weighted MASV-SMOTE (weighted distance metric)—and evaluate them under a 
strict zero data leakage nested cross-validation protocol.

Our empirical benchmark demonstrates that leaked SMOTE pipelines artificially inflate PR-AUC by 
up to +128%, and that SHAP-guided oversampling achieves superior minority recall, F1-score, 
and attribution stability compared to conventional SMOTE variants on the Kaggle Stroke Prediction 
dataset (19.5:1 imbalance ratio).`;

export default function PaperSection() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="paper" className="py-24 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <span className="badge bg-violet-500/20 text-violet-300 border border-violet-500/30 mb-4">
            📄 The Paper
          </span>
          <h2 className="section-heading">Research Paper</h2>
          <p className="section-subheading">
            Full paper available for download. Cite using the BibTeX entry below.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-10">
          {/* Abstract */}
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-bold text-white mb-4">Abstract</h3>
              <div className="text-sm text-slate-400 leading-relaxed whitespace-pre-line">
                {abstract}
              </div>
            </div>

            {/* Key stats */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Dataset", value: "Kaggle Stroke", sub: "fedesoriano/stroke-prediction-dataset" },
                { label: "Imbalance Ratio", value: "19.5:1", sub: "Only 4.87% stroke cases" },
                { label: "Evaluation", value: "5-Fold CV", sub: "Stratified Nested CV" },
                { label: "Algorithms", value: "9 Methods", sub: "Benchmarked end-to-end" },
              ].map((item) => (
                <div key={item.label} className="card text-center py-4">
                  <div className="text-sm text-slate-500 mb-1">{item.label}</div>
                  <div className="text-lg font-bold gradient-text">{item.value}</div>
                  <div className="text-xs text-slate-600 mt-1">{item.sub}</div>
                </div>
              ))}
            </div>

            {/* Download button */}
            <a
              href="/revised_shap_smote_paper.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-primary w-full justify-center text-base py-4"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Full Paper (PDF)
            </a>
          </div>

          {/* BibTeX + Research questions */}
          <div className="space-y-6">
            {/* BibTeX */}
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base font-bold text-white">Cite This Work</h3>
                <button
                  onClick={handleCopy}
                  className={`btn-outline text-xs py-1.5 px-3 ${copied ? "border-highlight text-highlight" : ""}`}
                >
                  {copied ? "✓ Copied!" : "Copy BibTeX"}
                </button>
              </div>
              <pre className="font-mono text-xs text-indigo-300 bg-slate-900 rounded-xl p-4 overflow-x-auto leading-relaxed border border-slate-800">
                {bibtex}
              </pre>
            </div>

            {/* Research questions */}
            <div className="card">
              <h3 className="text-base font-bold text-white mb-4">Research Questions</h3>
              <div className="space-y-3">
                {[
                  {
                    rq: "RQ1",
                    q: "What are the geometric limitations of Euclidean SMOTE, and how can XAI be taxonomized into data balancing?",
                    obj: "Objective 1",
                  },
                  {
                    rq: "RQ2",
                    q: "How can SHAP attributions guide synthetic generation via feature pruning vs. weighted distance metrics?",
                    obj: "Objective 2",
                  },
                  {
                    rq: "RQ3",
                    q: "Does SHAP-guided oversampling yield statistically significant improvements without data leakage?",
                    obj: "Objective 3",
                  },
                  {
                    rq: "RQ4",
                    q: "How does SHAP-guided oversampling impact attribution stability, calibration, and fidelity?",
                    obj: "Objective 4",
                  },
                ].map((item) => (
                  <div key={item.rq} className="flex gap-3">
                    <span className="flex-shrink-0 badge bg-primary/20 text-primary border-primary/30 self-start mt-0.5">
                      {item.rq}
                    </span>
                    <div>
                      <p className="text-sm text-slate-300">{item.q}</p>
                      <span className="text-xs text-slate-600">{item.obj}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
