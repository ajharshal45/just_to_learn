import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const snippets = [
  {
    id: "weighted",
    label: "SHAP-Weighted SMOTE",
    lang: "python",
    code: `from src.oversampling import SHAPWeightedSMOTE
from src.pipeline.datasets import load_benchmark_dataset

# Load imbalanced dataset
X, y, meta = load_benchmark_dataset("kaggle_stroke", random_state=42)
print(f"Imbalance ratio: {meta['imbalance_ratio']}")  # 19:1

# Fit and resample with SHAP-Weighted MASV-SMOTE
sampler = SHAPWeightedSMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"MASV Feature Weights: {sampler.feature_weights_}")`,
  },
  {
    id: "filtered",
    label: "SHAP-Filtered SMOTE",
    lang: "python",
    code: `from src.oversampling import SHAPFilteredSMOTE

# Use cumulative 90% MASV threshold for feature selection
sampler = SHAPFilteredSMOTE(
    cumulative_threshold=0.90,  # Keep top features covering 90% MASV
    k_neighbors=5,
    random_state=42,
)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Check which features were retained
print(f"Selected features: {sampler.selected_feature_indices_}")
print(f"Features kept: {len(sampler.selected_feature_indices_)}/{X.shape[1]}")`,
  },
  {
    id: "benchmark",
    label: "Leak-Free Benchmark",
    lang: "python",
    code: `from sklearn.ensemble import RandomForestClassifier
from src.pipeline.leak_free_cv import run_leak_free_benchmark

sampler = SHAPWeightedSMOTE(random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Run 5-fold stratified nested cross-validation
# All scaling, SHAP, and oversampling happen INSIDE each fold
results = run_leak_free_benchmark(
    X, y,
    oversampler=sampler,
    classifier=clf,
    n_splits=5,
    random_state=42,
)

print(f"PR-AUC:       {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
print(f"Minority F1:  {results['minority_f1_mean']:.4f}")
print(f"ECE:          {results['ece_mean']:.4f}")
print(f"Kendall τ:    {results['attribution_kendall_tau_mean']:.4f}")`,
  },
  {
    id: "cli",
    label: "CLI Benchmark",
    lang: "bash",
    code: `# Quick smoke test (fast)
python experiments/run_benchmark.py --quick

# Full benchmark suite (all datasets + classifiers)
python experiments/run_benchmark.py \\
  --config experiments/configs/benchmark_default.yaml

# Outputs:
#   experiments/results/benchmark_results.csv
#   experiments/results/benchmark_results.json
#   experiments/results/benchmark_results.tex  (LaTeX)
#   experiments/results/BENCHMARK_REPORT.md
#   experiments/results/figures/*.png`,
  },
];

export default function CodeSnippet() {
  const [active, setActive] = useState("weighted");
  const [copied, setCopied] = useState(false);

  const snippet = snippets.find((s) => s.id === active);

  const handleCopy = () => {
    navigator.clipboard.writeText(snippet.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="code" className="py-24 px-4 bg-slate-900/30">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <span className="badge bg-highlight/20 text-highlight border border-highlight/30 mb-4">
            💻 Quickstart
          </span>
          <h2 className="section-heading">Get Started in 3 Lines</h2>
          <p className="section-subheading">
            Install dependencies and start using SHAP-guided oversampling in your own pipelines.
          </p>
        </div>

        {/* Install */}
        <div className="card mb-8 border-slate-700">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Installation</h3>
          <div className="font-mono text-sm bg-slate-900 rounded-xl p-4 border border-slate-800 text-highlight overflow-x-auto">
            <span className="text-slate-600 select-none">$ </span>git clone https://github.com/hasanr-26/SHAP-SMOTE.git<br />
            <span className="text-slate-600 select-none">$ </span>cd SHAP-SMOTE<br />
            <span className="text-slate-600 select-none">$ </span>pip install -r requirements.txt
          </div>
        </div>

        {/* Code snippets */}
        <div className="card p-0 overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-slate-700 bg-slate-900/50 overflow-x-auto">
            {snippets.map((s) => (
              <button
                key={s.id}
                onClick={() => setActive(s.id)}
                className={`px-5 py-3 text-sm font-medium whitespace-nowrap transition-colors ${
                  active === s.id
                    ? "text-white border-b-2 border-primary bg-slate-800/50"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {s.label}
              </button>
            ))}
            <div className="ml-auto flex items-center px-4">
              <button
                onClick={handleCopy}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  copied
                    ? "border-highlight text-highlight"
                    : "border-slate-700 text-slate-500 hover:text-white hover:border-slate-500"
                }`}
              >
                {copied ? "✓ Copied" : "Copy"}
              </button>
            </div>
          </div>

          {/* Code */}
          <SyntaxHighlighter
            language={snippet.lang}
            style={vscDarkPlus}
            customStyle={{
              margin: 0,
              borderRadius: 0,
              background: "#0b1120",
              padding: "24px",
              fontSize: "13px",
              lineHeight: "1.7",
            }}
            showLineNumbers
          >
            {snippet.code}
          </SyntaxHighlighter>
        </div>
      </div>
    </section>
  );
}
