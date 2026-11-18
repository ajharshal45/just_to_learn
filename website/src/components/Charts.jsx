import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend,
} from "recharts";
import { globalResults, categoryColors } from "../data/benchmarkData";

const shortName = (name) =>
  name
    .replace("SHAP-Weighted MASV", "SHAP-Weighted")
    .replace("SHAP-Filtered", "SHAP-Filtered")
    .replace("Random Oversampling", "ROS")
    .replace("Raw (No Balancing)", "Raw")
    .replace("Standard SMOTE", "SMOTE")
    .replace("Borderline-SMOTE", "Borderline")
    .replace("SMOTE-ENN", "SMOTE-ENN")
    .replace("SMOTE-Tomek", "SMOTE-Tomek");

const chartTabs = [
  { id: "prAuc", label: "PR-AUC", metric: "prAuc", title: "PR-AUC (Average Precision) by Oversampler" },
  { id: "f1", label: "F1 Score", metric: "f1", title: "F1 Score by Oversampler" },
  { id: "ece", label: "Calibration (ECE)", metric: "ece", title: "Expected Calibration Error (lower = better)" },
  { id: "tau", label: "Attribution Stability", metric: "kendallTau", title: "Kendall's τ Feature Rank Stability" },
  { id: "radar", label: "Radar Overview", metric: null, title: "Multi-Dimensional Radar Comparison" },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm shadow-xl">
        <div className="font-semibold text-white mb-1">{label}</div>
        {payload.map((p) => (
          <div key={p.dataKey} className="text-slate-300">
            {p.name}: <span className="text-primary font-mono">{p.value?.toFixed(4)}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

function BarView({ metric, title, lowerBetter }) {
  const chartData = globalResults.map((r) => ({
    name: shortName(r.name),
    value: r[metric],
    category: r.category,
  }));

  return (
    <div>
      <h3 className="text-base font-semibold text-slate-300 mb-6">{title}</h3>
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="name"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            angle={-35}
            textAnchor="end"
            interval={0}
          />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="value"
            name={metric}
            radius={[6, 6, 0, 0]}
            fill="#6366f1"
            label={false}
          >
            {chartData.map((entry, index) => (
              <rect
                key={index}
                fill={categoryColors[entry.category] || "#6366f1"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-600 text-center mt-2">
        {lowerBetter ? "↓ Lower is better" : "↑ Higher is better"}
      </p>
    </div>
  );
}

function RadarView() {
  const radarData = [
    { metric: "PR-AUC", "SHAP-Weighted": 0.5667, "SMOTE": 0.5568, "Raw": 0.6087 },
    { metric: "F1", "SHAP-Weighted": 0.5284, "SMOTE": 0.5218, "Raw": 0.4269 },
    { metric: "Bal Acc", "SHAP-Weighted": 0.7754, "SMOTE": 0.7785, "Raw": 0.6819 },
    { metric: "KS Pass", "SHAP-Weighted": 0.9929, "SMOTE": 1.0, "Raw": 1.0 },
    { metric: "Stability τ", "SHAP-Weighted": 0.3394, "SMOTE": 0.3457, "Raw": 0.3591 },
  ];

  return (
    <div>
      <h3 className="text-base font-semibold text-slate-300 mb-6">
        Multi-Dimensional Radar: SHAP-Weighted vs SMOTE vs Raw
      </h3>
      <ResponsiveContainer width="100%" height={360}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#1e293b" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <Radar
            name="SHAP-Weighted MASV"
            dataKey="SHAP-Weighted"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Radar
            name="Standard SMOTE"
            dataKey="SMOTE"
            stroke="#6366f1"
            fill="#6366f1"
            fillOpacity={0.15}
            strokeWidth={2}
          />
          <Radar
            name="Raw (No Balancing)"
            dataKey="Raw"
            stroke="#64748b"
            fill="#64748b"
            fillOpacity={0.1}
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
          <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Charts() {
  const [activeTab, setActiveTab] = useState("prAuc");

  const active = chartTabs.find((t) => t.id === activeTab);

  return (
    <section id="charts" className="py-24 px-4 bg-slate-900/30">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <span className="badge bg-primary/20 text-primary border border-primary/30 mb-4">
            Charts & Visualizations
          </span>
          <h2 className="section-heading">Performance Charts</h2>
          <p className="section-subheading">
            Switch between metrics to explore how each oversampler performs across all evaluation dimensions.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex flex-wrap gap-2 mb-8">
          {chartTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-xl text-sm font-semibold transition-colors ${
                activeTab === tab.id
                  ? "bg-primary text-white"
                  : "bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Chart area */}
        <div className="card">
          {activeTab === "radar" ? (
            <RadarView />
          ) : (
            <BarView
              metric={active.metric}
              title={active.title}
              lowerBetter={activeTab === "ece"}
            />
          )}
        </div>

        {/* Proof figures grid */}
        <div className="mt-12">
          <h3 className="text-xl font-bold text-white mb-6">Publication Figures</h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { src: "/assets/proofs/fig1_geometric_decision_boundary_proof.png", caption: "Fig 1: Geometric Decision Boundary Proof" },
              { src: "/assets/proofs/fig3_weighting_schemes_comparison.png", caption: "Fig 3: Weighting Schemes Comparison (SHAP vs Gini vs MI)" },
              { src: "/assets/proofs/fig4_kaggle_stroke_pr_calibration_curves.png", caption: "Fig 4: Kaggle Stroke PR & Calibration Curves" },
              { src: "/assets/proofs/fig5_statistical_significance_and_stability.png", caption: "Fig 5: Statistical Significance & Stability" },
              { src: "/assets/figures/pr_auc_comparison.png", caption: "PR-AUC Comparison (All Datasets)" },
              { src: "/assets/figures/multi_dimensional_radar.png", caption: "Multi-Dimensional Radar Chart" },
            ].map((fig) => (
              <div key={fig.src} className="card p-3 group cursor-zoom-in">
                <img
                  src={fig.src}
                  alt={fig.caption}
                  className="w-full rounded-xl group-hover:scale-[1.02] transition-transform duration-200"
                  loading="lazy"
                />
                <p className="text-xs text-slate-500 text-center mt-2">{fig.caption}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
