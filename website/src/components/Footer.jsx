export default function Footer() {
  return (
    <footer className="py-16 px-4 border-t border-slate-800">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-10 mb-10">
          {/* Brand */}
          <div>
            <div className="text-xl font-black gradient-text mb-2">SHAP-SMOTE</div>
            <p className="text-slate-500 text-sm leading-relaxed">
              Explainability-Aware Synthetic Oversampling for Imbalanced Tabular Classification.
              A systematic benchmark of SHAP-Guided SMOTE methods under zero data leakage.
            </p>
            <div className="flex gap-3 mt-4">
              <span className="badge bg-slate-800 text-slate-400 border-slate-700">MIT License</span>
              <span className="badge bg-primary/20 text-primary border-primary/30">Python 3.10+</span>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Resources</h4>
            <div className="space-y-2">
              {[
                { href: "https://github.com/hasanr-26/SHAP-SMOTE", label: "GitHub Repository" },
                { href: "/revised_shap_smote_paper.pdf", label: "Research Paper (PDF)" },
                { href: "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset", label: "Kaggle Stroke Dataset" },
                { href: "https://github.com/hasanr-26/SHAP-SMOTE/blob/main/requirements.txt", label: "requirements.txt" },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-sm text-slate-500 hover:text-primary transition-colors"
                >
                  → {link.label}
                </a>
              ))}
            </div>
          </div>

          {/* Key citations */}
          <div>
            <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Key References</h4>
            <div className="space-y-2 text-xs text-slate-600">
              <p>Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. <em>JAIR</em>.</p>
              <p>Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. <em>NeurIPS</em>.</p>
              <p>He & Garcia (2009). Learning from Imbalanced Data. <em>IEEE TKDE</em>.</p>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-6 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-600">
          <p>© 2024 SHAP-SMOTE Research Project. MIT License.</p>
          <p>Built with React + Tailwind CSS + Recharts</p>
        </div>
      </div>
    </footer>
  );
}
