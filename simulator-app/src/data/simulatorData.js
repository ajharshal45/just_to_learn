// Comprehensive Simulator Dataset Registry with Real Benchmark Metrics

export const SIMULATOR_DATASETS = {
  "kaggle_stroke": {
    "id": "kaggle_stroke",
    "name": "Kaggle Stroke Prediction",
    "icon": "\ud83c\udfe5",
    "domain": "Clinical Healthcare / Neurology",
    "description": "Real patient records from Kaggle fedesoriano/stroke-prediction-dataset. Severe 19.5:1 imbalance where stroke cases represent only 4.87% of observations.",
    "imbalance_ratio": "19.5:1",
    "n_samples": 5110,
    "n_features": 11,
    "minority_pct": 4.87,
    "key_features": [
      "age",
      "avg_glucose_level",
      "hypertension",
      "heart_disease",
      "bmi"
    ],
    "noise_features": [
      "work_type_Govt",
      "work_type_Never_worked",
      "Residence_type_Urban"
    ],
    "shap_insight": "Age and average glucose level contribute 73.8% of the global MASV attribution. Standard SMOTE is distorted by one-hot encoded administrative columns (work_type, residence), whereas SHAP-Weighted SMOTE compresses them to retain true physiological clusters.",
    "methods": {
      "Raw (No Balancing)": {
        "random_forest": {
          "pr_auc": 0.1834,
          "recall": 0.082,
          "precision": 0.52,
          "f1": 0.1412,
          "bal_acc": 0.5401,
          "ece": 0.0382,
          "tau": 0.8333,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.179,
          "recall": 0.095,
          "precision": 0.48,
          "f1": 0.158,
          "bal_acc": 0.545,
          "ece": 0.041,
          "tau": 0.812,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "logistic_regression": {
          "pr_auc": 0.165,
          "recall": 0.05,
          "precision": 0.35,
          "f1": 0.0875,
          "bal_acc": 0.52,
          "ece": 0.045,
          "tau": 0.795,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        }
      },
      "Standard SMOTE": {
        "random_forest": {
          "pr_auc": 0.1742,
          "recall": 0.2854,
          "precision": 0.1678,
          "f1": 0.211,
          "bal_acc": 0.612,
          "ece": 0.1624,
          "tau": 0.8481,
          "wasserstein": 2.14,
          "ks_pass": 95.5
        },
        "xgboost": {
          "pr_auc": 0.171,
          "recall": 0.31,
          "precision": 0.152,
          "f1": 0.204,
          "bal_acc": 0.621,
          "ece": 0.172,
          "tau": 0.825,
          "wasserstein": 2.14,
          "ks_pass": 95.5
        },
        "logistic_regression": {
          "pr_auc": 0.158,
          "recall": 0.68,
          "precision": 0.085,
          "f1": 0.1511,
          "bal_acc": 0.71,
          "ece": 0.284,
          "tau": 0.78,
          "wasserstein": 2.14,
          "ks_pass": 95.5
        }
      },
      "Borderline-SMOTE": {
        "random_forest": {
          "pr_auc": 0.1765,
          "recall": 0.2714,
          "precision": 0.1689,
          "f1": 0.2084,
          "bal_acc": 0.608,
          "ece": 0.1481,
          "tau": 0.7913,
          "wasserstein": 7.3981,
          "ks_pass": 82.2
        },
        "xgboost": {
          "pr_auc": 0.173,
          "recall": 0.295,
          "precision": 0.159,
          "f1": 0.2067,
          "bal_acc": 0.615,
          "ece": 0.156,
          "tau": 0.78,
          "wasserstein": 7.3981,
          "ks_pass": 82.2
        },
        "logistic_regression": {
          "pr_auc": 0.161,
          "recall": 0.64,
          "precision": 0.089,
          "f1": 0.1563,
          "bal_acc": 0.698,
          "ece": 0.265,
          "tau": 0.76,
          "wasserstein": 7.3981,
          "ks_pass": 82.2
        }
      },
      "ADASYN": {
        "random_forest": {
          "pr_auc": 0.171,
          "recall": 0.298,
          "precision": 0.152,
          "f1": 0.2014,
          "bal_acc": 0.615,
          "ece": 0.173,
          "tau": 0.784,
          "wasserstein": 7.6391,
          "ks_pass": 88.3
        },
        "xgboost": {
          "pr_auc": 0.168,
          "recall": 0.32,
          "precision": 0.141,
          "f1": 0.1957,
          "bal_acc": 0.623,
          "ece": 0.181,
          "tau": 0.77,
          "wasserstein": 7.6391,
          "ks_pass": 88.3
        },
        "logistic_regression": {
          "pr_auc": 0.154,
          "recall": 0.71,
          "precision": 0.081,
          "f1": 0.1454,
          "bal_acc": 0.715,
          "ece": 0.299,
          "tau": 0.75,
          "wasserstein": 7.6391,
          "ks_pass": 88.3
        }
      },
      "SHAP-Filtered SMOTE": {
        "random_forest": {
          "pr_auc": 0.1812,
          "recall": 0.2921,
          "precision": 0.1788,
          "f1": 0.2215,
          "bal_acc": 0.622,
          "ece": 0.158,
          "tau": 0.72,
          "wasserstein": 1.6249,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.1795,
          "recall": 0.315,
          "precision": 0.169,
          "f1": 0.2201,
          "bal_acc": 0.629,
          "ece": 0.164,
          "tau": 0.715,
          "wasserstein": 1.6249,
          "ks_pass": 100.0
        },
        "logistic_regression": {
          "pr_auc": 0.162,
          "recall": 0.69,
          "precision": 0.088,
          "f1": 0.1561,
          "bal_acc": 0.718,
          "ece": 0.278,
          "tau": 0.71,
          "wasserstein": 1.6249,
          "ks_pass": 100.0
        }
      },
      "SHAP-Weighted MASV-SMOTE": {
        "random_forest": {
          "pr_auc": 0.1865,
          "recall": 0.315,
          "precision": 0.1862,
          "f1": 0.2341,
          "bal_acc": 0.638,
          "ece": 0.1512,
          "tau": 0.837,
          "wasserstein": 1.6422,
          "ks_pass": 99.3
        },
        "xgboost": {
          "pr_auc": 0.184,
          "recall": 0.338,
          "precision": 0.179,
          "f1": 0.234,
          "bal_acc": 0.645,
          "ece": 0.159,
          "tau": 0.829,
          "wasserstein": 1.6422,
          "ks_pass": 99.3
        },
        "logistic_regression": {
          "pr_auc": 0.168,
          "recall": 0.71,
          "precision": 0.092,
          "f1": 0.1629,
          "bal_acc": 0.725,
          "ece": 0.271,
          "tau": 0.81,
          "wasserstein": 1.6422,
          "ks_pass": 99.3
        }
      }
    }
  },
  "credit_fraud": {
    "id": "credit_fraud",
    "name": "Credit Card Fraud Simulation",
    "icon": "\ud83d\udcb3",
    "domain": "Financial Transaction Security",
    "description": "High-dimensional PCA components with extreme 49:1 imbalance (2% fraud transactions). Fraud instances shift along latent axes V14, V10, V12, V4.",
    "imbalance_ratio": "49:1",
    "n_samples": 2000,
    "n_features": 28,
    "minority_pct": 2.0,
    "key_features": [
      "V14",
      "V10",
      "V12",
      "V4",
      "V11"
    ],
    "noise_features": [
      "V22",
      "V23",
      "V24",
      "V25",
      "V28"
    ],
    "shap_insight": "PCA transaction variables contain sparse non-linear signatures. SHAP-Weighted SMOTE focuses nearest-neighbour interpolation on principal components V14 & V10 while treating uninformative tail components as negligible, achieving +0.028 PR-AUC improvement.",
    "methods": {
      "Raw (No Balancing)": {
        "logistic_regression": {
          "pr_auc": 0.1736,
          "recall": 0.025,
          "precision": 0.2,
          "f1": 0.0444,
          "bal_acc": 0.5125,
          "ece": 0.0053,
          "tau": 0.2698,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9708,
          "recall": 0.675,
          "precision": 0.9667,
          "f1": 0.7934,
          "bal_acc": 0.8372,
          "ece": 0.0139,
          "tau": 0.5799,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9808,
          "recall": 0.725,
          "precision": 1.0,
          "f1": 0.8372,
          "bal_acc": 0.8625,
          "ece": 0.0059,
          "tau": 0.0,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        }
      },
      "Random Oversampling (ROS)": {
        "logistic_regression": {
          "pr_auc": 0.0626,
          "recall": 0.4,
          "precision": 0.0302,
          "f1": 0.0561,
          "bal_acc": 0.5747,
          "ece": 0.2989,
          "tau": 0.1386,
          "wasserstein": 0.0463,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9402,
          "recall": 0.45,
          "precision": 1.0,
          "f1": 0.601,
          "bal_acc": 0.725,
          "ece": 0.0301,
          "tau": 0.564,
          "wasserstein": 0.0463,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9865,
          "recall": 0.95,
          "precision": 0.92,
          "f1": 0.9289,
          "bal_acc": 0.974,
          "ece": 0.0057,
          "tau": 0.0,
          "wasserstein": 0.0463,
          "ks_pass": 100.0
        }
      },
      "Standard SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.0915,
          "recall": 0.425,
          "precision": 0.0372,
          "f1": 0.0683,
          "bal_acc": 0.6036,
          "ece": 0.2635,
          "tau": 0.1884,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.8756,
          "recall": 0.85,
          "precision": 0.7195,
          "f1": 0.7686,
          "bal_acc": 0.9212,
          "ece": 0.0788,
          "tau": 0.6063,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9762,
          "recall": 0.95,
          "precision": 0.8448,
          "f1": 0.8836,
          "bal_acc": 0.9727,
          "ece": 0.0111,
          "tau": 0.0,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        }
      },
      "Borderline-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.2497,
          "recall": 0.325,
          "precision": 0.1107,
          "f1": 0.1644,
          "bal_acc": 0.6367,
          "ece": 0.071,
          "tau": 0.2593,
          "wasserstein": 0.4126,
          "ks_pass": 82.1
        },
        "random_forest": {
          "pr_auc": 0.8814,
          "recall": 0.5,
          "precision": 0.91,
          "f1": 0.6436,
          "bal_acc": 0.7495,
          "ece": 0.0384,
          "tau": 0.636,
          "wasserstein": 0.4126,
          "ks_pass": 82.1
        },
        "xgboost": {
          "pr_auc": 0.9332,
          "recall": 0.85,
          "precision": 0.8239,
          "f1": 0.8301,
          "bal_acc": 0.923,
          "ece": 0.0093,
          "tau": 0.0,
          "wasserstein": 0.4126,
          "ks_pass": 82.1
        }
      },
      "ADASYN": {
        "logistic_regression": {
          "pr_auc": 0.081,
          "recall": 0.375,
          "precision": 0.033,
          "f1": 0.0606,
          "bal_acc": 0.5776,
          "ece": 0.2618,
          "tau": 0.1915,
          "wasserstein": 0.2598,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.8964,
          "recall": 0.775,
          "precision": 0.7371,
          "f1": 0.7373,
          "bal_acc": 0.8842,
          "ece": 0.0825,
          "tau": 0.6021,
          "wasserstein": 0.2598,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9814,
          "recall": 0.95,
          "precision": 0.8359,
          "f1": 0.8801,
          "bal_acc": 0.9727,
          "ece": 0.0104,
          "tau": 0.0,
          "wasserstein": 0.2598,
          "ks_pass": 100.0
        }
      },
      "SMOTE-ENN": {
        "logistic_regression": {
          "pr_auc": 0.0926,
          "recall": 0.425,
          "precision": 0.0372,
          "f1": 0.0683,
          "bal_acc": 0.6036,
          "ece": 0.2636,
          "tau": 0.1926,
          "wasserstein": 0.2677,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.8826,
          "recall": 0.775,
          "precision": 0.8,
          "f1": 0.7771,
          "bal_acc": 0.8855,
          "ece": 0.0797,
          "tau": 0.6233,
          "wasserstein": 0.2677,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.978,
          "recall": 0.925,
          "precision": 0.8421,
          "f1": 0.8704,
          "bal_acc": 0.9602,
          "ece": 0.0105,
          "tau": 0.0,
          "wasserstein": 0.2677,
          "ks_pass": 100.0
        }
      },
      "SMOTE-Tomek": {
        "logistic_regression": {
          "pr_auc": 0.0915,
          "recall": 0.425,
          "precision": 0.0372,
          "f1": 0.0683,
          "bal_acc": 0.6036,
          "ece": 0.2635,
          "tau": 0.1884,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.8756,
          "recall": 0.85,
          "precision": 0.7195,
          "f1": 0.7686,
          "bal_acc": 0.9212,
          "ece": 0.0788,
          "tau": 0.6063,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9762,
          "recall": 0.95,
          "precision": 0.8448,
          "f1": 0.8836,
          "bal_acc": 0.9727,
          "ece": 0.0111,
          "tau": 0.0,
          "wasserstein": 0.2676,
          "ks_pass": 100.0
        }
      },
      "SHAP-Filtered SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.0853,
          "recall": 0.4,
          "precision": 0.0345,
          "f1": 0.0635,
          "bal_acc": 0.5906,
          "ece": 0.2614,
          "tau": 0.1831,
          "wasserstein": 0.2578,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.8762,
          "recall": 0.775,
          "precision": 0.8212,
          "f1": 0.7663,
          "bal_acc": 0.8852,
          "ece": 0.0808,
          "tau": 0.6,
          "wasserstein": 0.2578,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9649,
          "recall": 0.975,
          "precision": 0.7898,
          "f1": 0.8657,
          "bal_acc": 0.9844,
          "ece": 0.0109,
          "tau": 0.0,
          "wasserstein": 0.2578,
          "ks_pass": 100.0
        }
      },
      "SHAP-Weighted MASV-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.0832,
          "recall": 0.375,
          "precision": 0.0323,
          "f1": 0.0595,
          "bal_acc": 0.5758,
          "ece": 0.266,
          "tau": 0.1862,
          "wasserstein": 0.2583,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.904,
          "recall": 0.75,
          "precision": 0.8229,
          "f1": 0.7644,
          "bal_acc": 0.873,
          "ece": 0.0749,
          "tau": 0.6138,
          "wasserstein": 0.2583,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9738,
          "recall": 0.95,
          "precision": 0.8683,
          "f1": 0.8994,
          "bal_acc": 0.9732,
          "ece": 0.0103,
          "tau": 0.0,
          "wasserstein": 0.2583,
          "ks_pass": 100.0
        }
      }
    }
  },
  "synthetic": {
    "id": "synthetic",
    "name": "Synthetic Non-Linear Manifold",
    "icon": "\ud83e\uddea",
    "domain": "Controlled Tabular Simulation",
    "description": "Controlled 18:1 imbalance with 6 true informative dimensions, 4 redundant correlates, and 10 pure Gaussian noise features.",
    "imbalance_ratio": "18:1",
    "n_samples": 1000,
    "n_features": 20,
    "minority_pct": 5.0,
    "key_features": [
      "Feature_01",
      "Feature_03",
      "Feature_05",
      "Feature_08"
    ],
    "noise_features": [
      "Feature_11",
      "Feature_14",
      "Feature_17",
      "Feature_19"
    ],
    "shap_insight": "Pure noise dimensions corrupt standard Euclidean distance by 50% of total distance magnitude. SHAP-Filtered SMOTE prunes them entirely, boosting PR-AUC from 0.3765 to 0.4187 (+11.2%).",
    "methods": {
      "Raw (No Balancing)": {
        "logistic_regression": {
          "pr_auc": 0.3852,
          "recall": 0.1473,
          "precision": 0.6524,
          "f1": 0.2239,
          "bal_acc": 0.5699,
          "ece": 0.0282,
          "tau": 0.6126,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.5463,
          "recall": 0.0909,
          "precision": 0.8,
          "f1": 0.1615,
          "bal_acc": 0.5455,
          "ece": 0.0236,
          "tau": 0.6884,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5284,
          "recall": 0.2036,
          "precision": 0.7267,
          "f1": 0.3117,
          "bal_acc": 0.5997,
          "ece": 0.0284,
          "tau": 0.0,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        }
      },
      "Random Oversampling (ROS)": {
        "logistic_regression": {
          "pr_auc": 0.3413,
          "recall": 0.6709,
          "precision": 0.153,
          "f1": 0.2487,
          "bal_acc": 0.7308,
          "ece": 0.2309,
          "tau": 0.5347,
          "wasserstein": 0.0538,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.4362,
          "recall": 0.3527,
          "precision": 0.3745,
          "f1": 0.3613,
          "bal_acc": 0.66,
          "ece": 0.1035,
          "tau": 0.6211,
          "wasserstein": 0.0538,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5238,
          "recall": 0.3527,
          "precision": 0.6479,
          "f1": 0.4512,
          "bal_acc": 0.6705,
          "ece": 0.0272,
          "tau": 0.0,
          "wasserstein": 0.0538,
          "ks_pass": 100.0
        }
      },
      "Standard SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.3579,
          "recall": 0.6709,
          "precision": 0.159,
          "f1": 0.2564,
          "bal_acc": 0.7356,
          "ece": 0.2119,
          "tau": 0.4716,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.3765,
          "recall": 0.6127,
          "precision": 0.3055,
          "f1": 0.4059,
          "bal_acc": 0.7651,
          "ece": 0.1494,
          "tau": 0.6589,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5359,
          "recall": 0.5564,
          "precision": 0.4983,
          "f1": 0.5239,
          "bal_acc": 0.7618,
          "ece": 0.039,
          "tau": 0.0,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        }
      },
      "Borderline-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.3566,
          "recall": 0.5964,
          "precision": 0.2087,
          "f1": 0.3077,
          "bal_acc": 0.73,
          "ece": 0.1453,
          "tau": 0.5663,
          "wasserstein": 0.2136,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.4518,
          "recall": 0.4836,
          "precision": 0.4346,
          "f1": 0.4537,
          "bal_acc": 0.7233,
          "ece": 0.0993,
          "tau": 0.6611,
          "wasserstein": 0.2136,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5915,
          "recall": 0.4818,
          "precision": 0.52,
          "f1": 0.4985,
          "bal_acc": 0.7282,
          "ece": 0.0374,
          "tau": 0.0,
          "wasserstein": 0.2136,
          "ks_pass": 100.0
        }
      },
      "ADASYN": {
        "logistic_regression": {
          "pr_auc": 0.3368,
          "recall": 0.6145,
          "precision": 0.1482,
          "f1": 0.2379,
          "bal_acc": 0.7074,
          "ece": 0.217,
          "tau": 0.4947,
          "wasserstein": 0.2103,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.3843,
          "recall": 0.5564,
          "precision": 0.3026,
          "f1": 0.3875,
          "bal_acc": 0.738,
          "ece": 0.1536,
          "tau": 0.6505,
          "wasserstein": 0.2103,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5345,
          "recall": 0.5,
          "precision": 0.4837,
          "f1": 0.4838,
          "bal_acc": 0.7325,
          "ece": 0.0455,
          "tau": 0.0,
          "wasserstein": 0.2103,
          "ks_pass": 100.0
        }
      },
      "SMOTE-ENN": {
        "logistic_regression": {
          "pr_auc": 0.351,
          "recall": 0.7455,
          "precision": 0.1512,
          "f1": 0.2505,
          "bal_acc": 0.7538,
          "ece": 0.2476,
          "tau": 0.4463,
          "wasserstein": 0.182,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.3626,
          "recall": 0.6691,
          "precision": 0.2485,
          "f1": 0.3601,
          "bal_acc": 0.7748,
          "ece": 0.1852,
          "tau": 0.6105,
          "wasserstein": 0.182,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5386,
          "recall": 0.6855,
          "precision": 0.3271,
          "f1": 0.4392,
          "bal_acc": 0.8004,
          "ece": 0.0842,
          "tau": 0.0,
          "wasserstein": 0.182,
          "ks_pass": 100.0
        }
      },
      "SMOTE-Tomek": {
        "logistic_regression": {
          "pr_auc": 0.3579,
          "recall": 0.6709,
          "precision": 0.159,
          "f1": 0.2564,
          "bal_acc": 0.7356,
          "ece": 0.2119,
          "tau": 0.4716,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.3765,
          "recall": 0.6127,
          "precision": 0.3055,
          "f1": 0.4059,
          "bal_acc": 0.7651,
          "ece": 0.1494,
          "tau": 0.6589,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5359,
          "recall": 0.5564,
          "precision": 0.4983,
          "f1": 0.5239,
          "bal_acc": 0.7618,
          "ece": 0.039,
          "tau": 0.0,
          "wasserstein": 0.174,
          "ks_pass": 100.0
        }
      },
      "SHAP-Filtered SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.3613,
          "recall": 0.6364,
          "precision": 0.1624,
          "f1": 0.2574,
          "bal_acc": 0.7257,
          "ece": 0.2059,
          "tau": 0.4926,
          "wasserstein": 0.1835,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.4187,
          "recall": 0.5564,
          "precision": 0.3395,
          "f1": 0.4156,
          "bal_acc": 0.7433,
          "ece": 0.1501,
          "tau": 0.6063,
          "wasserstein": 0.1835,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5398,
          "recall": 0.5382,
          "precision": 0.4919,
          "f1": 0.5114,
          "bal_acc": 0.7522,
          "ece": 0.0427,
          "tau": 0.0,
          "wasserstein": 0.1835,
          "ks_pass": 100.0
        }
      },
      "SHAP-Weighted MASV-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.3597,
          "recall": 0.6364,
          "precision": 0.1566,
          "f1": 0.2505,
          "bal_acc": 0.7225,
          "ece": 0.2108,
          "tau": 0.4884,
          "wasserstein": 0.1912,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.4104,
          "recall": 0.5745,
          "precision": 0.3188,
          "f1": 0.4078,
          "bal_acc": 0.7503,
          "ece": 0.1448,
          "tau": 0.6505,
          "wasserstein": 0.1912,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.5493,
          "recall": 0.5382,
          "precision": 0.4535,
          "f1": 0.4889,
          "bal_acc": 0.7501,
          "ece": 0.0431,
          "tau": 0.0,
          "wasserstein": 0.1912,
          "ks_pass": 100.0
        }
      }
    }
  },
  "nsl_kdd": {
    "id": "nsl_kdd",
    "name": "Network Intrusion (NSL-KDD)",
    "icon": "\ud83d\udee1\ufe0f",
    "domain": "Cybersecurity / Intrusion Detection",
    "description": "Multimodal cluster distributions with severe boundary overlap between attack signatures and legitimate network traffic.",
    "imbalance_ratio": "11.5:1",
    "n_samples": 1500,
    "n_features": 35,
    "minority_pct": 8.0,
    "key_features": [
      "dst_host_srv_count",
      "serror_rate",
      "src_bytes",
      "same_srv_rate"
    ],
    "noise_features": [
      "num_root",
      "num_file_creations",
      "is_guest_login"
    ],
    "shap_insight": "Severe boundary overlap causes Standard SMOTE to generate bridge instances inside normal network traffic. SHAP-SMOTE elevates minority F1 score from 0.2455 to 0.2940 (+19.7%).",
    "methods": {
      "Raw (No Balancing)": {
        "logistic_regression": {
          "pr_auc": 0.1559,
          "recall": 0.0,
          "precision": 0.0,
          "f1": 0.0,
          "bal_acc": 0.4989,
          "ece": 0.0404,
          "tau": 0.4071,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.3273,
          "recall": 0.0,
          "precision": 0.0,
          "f1": 0.0,
          "bal_acc": 0.5,
          "ece": 0.0205,
          "tau": 0.5072,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.3719,
          "recall": 0.0524,
          "precision": 0.6333,
          "f1": 0.0952,
          "bal_acc": 0.5247,
          "ece": 0.0453,
          "tau": 0.0,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        }
      },
      "Random Oversampling (ROS)": {
        "logistic_regression": {
          "pr_auc": 0.1357,
          "recall": 0.5339,
          "precision": 0.1248,
          "f1": 0.2023,
          "bal_acc": 0.5852,
          "ece": 0.3488,
          "tau": 0.4602,
          "wasserstein": 0.0641,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.2445,
          "recall": 0.3544,
          "precision": 0.2066,
          "f1": 0.26,
          "bal_acc": 0.611,
          "ece": 0.2102,
          "tau": 0.5516,
          "wasserstein": 0.0641,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.3703,
          "recall": 0.2179,
          "precision": 0.462,
          "f1": 0.287,
          "bal_acc": 0.5951,
          "ece": 0.0346,
          "tau": 0.0,
          "wasserstein": 0.0641,
          "ks_pass": 100.0
        }
      },
      "Standard SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.1333,
          "recall": 0.5108,
          "precision": 0.122,
          "f1": 0.197,
          "bal_acc": 0.5773,
          "ece": 0.3345,
          "tau": 0.4461,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.2154,
          "recall": 0.414,
          "precision": 0.1751,
          "f1": 0.2455,
          "bal_acc": 0.6122,
          "ece": 0.2419,
          "tau": 0.602,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.2598,
          "recall": 0.308,
          "precision": 0.2605,
          "f1": 0.2819,
          "bal_acc": 0.6108,
          "ece": 0.0848,
          "tau": 0.0,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        }
      },
      "Borderline-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.1403,
          "recall": 0.5105,
          "precision": 0.1271,
          "f1": 0.2036,
          "bal_acc": 0.5848,
          "ece": 0.3183,
          "tau": 0.3634,
          "wasserstein": 0.2456,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.1896,
          "recall": 0.4217,
          "precision": 0.1982,
          "f1": 0.2689,
          "bal_acc": 0.6278,
          "ece": 0.2287,
          "tau": 0.5758,
          "wasserstein": 0.2456,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.27,
          "recall": 0.2934,
          "precision": 0.2871,
          "f1": 0.2877,
          "bal_acc": 0.6102,
          "ece": 0.071,
          "tau": 0.0,
          "wasserstein": 0.2456,
          "ks_pass": 100.0
        }
      },
      "ADASYN": {
        "logistic_regression": {
          "pr_auc": 0.1444,
          "recall": 0.4957,
          "precision": 0.1196,
          "f1": 0.1927,
          "bal_acc": 0.5708,
          "ece": 0.3357,
          "tau": 0.444,
          "wasserstein": 0.2232,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.191,
          "recall": 0.4587,
          "precision": 0.1894,
          "f1": 0.2678,
          "bal_acc": 0.6335,
          "ece": 0.2475,
          "tau": 0.557,
          "wasserstein": 0.2232,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.2845,
          "recall": 0.3165,
          "precision": 0.2677,
          "f1": 0.2888,
          "bal_acc": 0.6158,
          "ece": 0.0871,
          "tau": 0.0,
          "wasserstein": 0.2232,
          "ks_pass": 100.0
        }
      },
      "SMOTE-ENN": {
        "logistic_regression": {
          "pr_auc": 0.1344,
          "recall": 0.7595,
          "precision": 0.1222,
          "f1": 0.2105,
          "bal_acc": 0.6142,
          "ece": 0.4493,
          "tau": 0.4669,
          "wasserstein": 0.2098,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.2034,
          "recall": 0.7895,
          "precision": 0.1639,
          "f1": 0.2712,
          "bal_acc": 0.699,
          "ece": 0.3615,
          "tau": 0.643,
          "wasserstein": 0.2098,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.2435,
          "recall": 0.6695,
          "precision": 0.2188,
          "f1": 0.3293,
          "bal_acc": 0.7184,
          "ece": 0.2163,
          "tau": 0.0,
          "wasserstein": 0.2098,
          "ks_pass": 100.0
        }
      },
      "SMOTE-Tomek": {
        "logistic_regression": {
          "pr_auc": 0.1333,
          "recall": 0.5108,
          "precision": 0.122,
          "f1": 0.197,
          "bal_acc": 0.5773,
          "ece": 0.3345,
          "tau": 0.4461,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.2154,
          "recall": 0.414,
          "precision": 0.1751,
          "f1": 0.2455,
          "bal_acc": 0.6122,
          "ece": 0.2419,
          "tau": 0.602,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.2598,
          "recall": 0.308,
          "precision": 0.2605,
          "f1": 0.2819,
          "bal_acc": 0.6108,
          "ece": 0.0848,
          "tau": 0.0,
          "wasserstein": 0.2069,
          "ks_pass": 100.0
        }
      },
      "SHAP-Filtered SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.1437,
          "recall": 0.5484,
          "precision": 0.1286,
          "f1": 0.2084,
          "bal_acc": 0.5939,
          "ece": 0.3347,
          "tau": 0.4017,
          "wasserstein": 0.2291,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.1982,
          "recall": 0.4442,
          "precision": 0.1861,
          "f1": 0.2621,
          "bal_acc": 0.6277,
          "ece": 0.2477,
          "tau": 0.5025,
          "wasserstein": 0.2291,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.2672,
          "recall": 0.3311,
          "precision": 0.2852,
          "f1": 0.3057,
          "bal_acc": 0.6253,
          "ece": 0.0886,
          "tau": 0.0,
          "wasserstein": 0.2291,
          "ks_pass": 100.0
        }
      },
      "SHAP-Weighted MASV-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.1443,
          "recall": 0.5188,
          "precision": 0.1265,
          "f1": 0.2034,
          "bal_acc": 0.5857,
          "ece": 0.3353,
          "tau": 0.405,
          "wasserstein": 0.2342,
          "ks_pass": 97.1
        },
        "random_forest": {
          "pr_auc": 0.2115,
          "recall": 0.4889,
          "precision": 0.2107,
          "f1": 0.294,
          "bal_acc": 0.6552,
          "ece": 0.2455,
          "tau": 0.5771,
          "wasserstein": 0.2342,
          "ks_pass": 97.1
        },
        "xgboost": {
          "pr_auc": 0.3019,
          "recall": 0.3148,
          "precision": 0.3109,
          "f1": 0.3111,
          "bal_acc": 0.6227,
          "ece": 0.0805,
          "tau": 0.0,
          "wasserstein": 0.2342,
          "ks_pass": 97.1
        }
      }
    }
  },
  "breast_cancer": {
    "id": "breast_cancer",
    "name": "Breast Cancer Diagnostic (WDBC)",
    "icon": "\ud83c\udf97\ufe0f",
    "domain": "Oncology Diagnostics",
    "description": "Wisconsin Diagnostic Breast Cancer dataset with malignant biopsy cases subsampled to 9:1 imbalance ratio.",
    "imbalance_ratio": "9:1",
    "n_samples": 397,
    "n_features": 30,
    "minority_pct": 10.0,
    "key_features": [
      "worst concave points",
      "worst radius",
      "mean concave points",
      "worst perimeter"
    ],
    "noise_features": [
      "symmetry error",
      "texture error",
      "smoothness error"
    ],
    "shap_insight": "High feature collinearity exists between mean, standard error, and worst cell nuclear dimensions. SHAP weights emphasize worst concave points, maintaining 0.9527 PR-AUC.",
    "methods": {
      "Raw (No Balancing)": {
        "logistic_regression": {
          "pr_auc": 0.9626,
          "recall": 0.85,
          "precision": 0.9714,
          "f1": 0.9048,
          "bal_acc": 0.9236,
          "ece": 0.0296,
          "tau": 0.4989,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9469,
          "recall": 0.8,
          "precision": 0.9714,
          "f1": 0.8743,
          "bal_acc": 0.8986,
          "ece": 0.0273,
          "tau": 0.7453,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9552,
          "recall": 0.825,
          "precision": 0.9464,
          "f1": 0.8762,
          "bal_acc": 0.9097,
          "ece": 0.0234,
          "tau": 0.0,
          "wasserstein": 0.0,
          "ks_pass": 100.0
        }
      },
      "Random Oversampling (ROS)": {
        "logistic_regression": {
          "pr_auc": 0.9643,
          "recall": 0.875,
          "precision": 0.8456,
          "f1": 0.8569,
          "bal_acc": 0.9277,
          "ece": 0.0471,
          "tau": 0.4032,
          "wasserstein": 3.4697,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9505,
          "recall": 0.825,
          "precision": 0.9464,
          "f1": 0.8779,
          "bal_acc": 0.9097,
          "ece": 0.0288,
          "tau": 0.7425,
          "wasserstein": 3.4697,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9499,
          "recall": 0.875,
          "precision": 0.8575,
          "f1": 0.8614,
          "bal_acc": 0.9291,
          "ece": 0.0288,
          "tau": 0.0,
          "wasserstein": 3.4697,
          "ks_pass": 100.0
        }
      },
      "Standard SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.9646,
          "recall": 0.875,
          "precision": 0.9306,
          "f1": 0.8978,
          "bal_acc": 0.9333,
          "ece": 0.0401,
          "tau": 0.417,
          "wasserstein": 4.4418,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.947,
          "recall": 0.85,
          "precision": 0.8806,
          "f1": 0.8595,
          "bal_acc": 0.918,
          "ece": 0.0386,
          "tau": 0.7582,
          "wasserstein": 4.4418,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9485,
          "recall": 0.875,
          "precision": 0.877,
          "f1": 0.873,
          "bal_acc": 0.9305,
          "ece": 0.0254,
          "tau": 0.0,
          "wasserstein": 4.4418,
          "ks_pass": 100.0
        }
      },
      "Borderline-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.9502,
          "recall": 0.875,
          "precision": 0.7828,
          "f1": 0.8237,
          "bal_acc": 0.9235,
          "ece": 0.051,
          "tau": 0.3683,
          "wasserstein": 28.7208,
          "ks_pass": 46.7
        },
        "random_forest": {
          "pr_auc": 0.938,
          "recall": 0.85,
          "precision": 0.7944,
          "f1": 0.8175,
          "bal_acc": 0.9124,
          "ece": 0.0464,
          "tau": 0.7057,
          "wasserstein": 28.7208,
          "ks_pass": 46.7
        },
        "xgboost": {
          "pr_auc": 0.9525,
          "recall": 0.875,
          "precision": 0.8206,
          "f1": 0.8422,
          "bal_acc": 0.9263,
          "ece": 0.032,
          "tau": 0.0,
          "wasserstein": 28.7208,
          "ks_pass": 46.7
        }
      },
      "ADASYN": {
        "logistic_regression": {
          "pr_auc": 0.9661,
          "recall": 0.9,
          "precision": 0.8206,
          "f1": 0.8569,
          "bal_acc": 0.9388,
          "ece": 0.049,
          "tau": 0.4087,
          "wasserstein": 29.8631,
          "ks_pass": 53.3
        },
        "random_forest": {
          "pr_auc": 0.9332,
          "recall": 0.875,
          "precision": 0.7917,
          "f1": 0.8294,
          "bal_acc": 0.9249,
          "ece": 0.044,
          "tau": 0.7701,
          "wasserstein": 29.8631,
          "ks_pass": 53.3
        },
        "xgboost": {
          "pr_auc": 0.9376,
          "recall": 0.85,
          "precision": 0.8714,
          "f1": 0.8583,
          "bal_acc": 0.918,
          "ece": 0.031,
          "tau": 0.0,
          "wasserstein": 29.8631,
          "ks_pass": 53.3
        }
      },
      "SMOTE-ENN": {
        "logistic_regression": {
          "pr_auc": 0.9585,
          "recall": 0.9,
          "precision": 0.8056,
          "f1": 0.848,
          "bal_acc": 0.9374,
          "ece": 0.0504,
          "tau": 0.3894,
          "wasserstein": 5.9036,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9374,
          "recall": 0.875,
          "precision": 0.7956,
          "f1": 0.8319,
          "bal_acc": 0.9249,
          "ece": 0.0449,
          "tau": 0.7628,
          "wasserstein": 5.9036,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9527,
          "recall": 0.9,
          "precision": 0.8288,
          "f1": 0.8579,
          "bal_acc": 0.9388,
          "ece": 0.0339,
          "tau": 0.0,
          "wasserstein": 5.9036,
          "ks_pass": 100.0
        }
      },
      "SMOTE-Tomek": {
        "logistic_regression": {
          "pr_auc": 0.9646,
          "recall": 0.875,
          "precision": 0.9306,
          "f1": 0.8978,
          "bal_acc": 0.9333,
          "ece": 0.0401,
          "tau": 0.417,
          "wasserstein": 4.8646,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.947,
          "recall": 0.85,
          "precision": 0.8806,
          "f1": 0.8595,
          "bal_acc": 0.918,
          "ece": 0.0386,
          "tau": 0.7582,
          "wasserstein": 4.8646,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9485,
          "recall": 0.875,
          "precision": 0.877,
          "f1": 0.873,
          "bal_acc": 0.9305,
          "ece": 0.0254,
          "tau": 0.0,
          "wasserstein": 4.8646,
          "ks_pass": 100.0
        }
      },
      "SHAP-Filtered SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.9668,
          "recall": 0.9,
          "precision": 0.8892,
          "f1": 0.8905,
          "bal_acc": 0.943,
          "ece": 0.0417,
          "tau": 0.3995,
          "wasserstein": 5.8291,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9527,
          "recall": 0.85,
          "precision": 0.8575,
          "f1": 0.8461,
          "bal_acc": 0.9166,
          "ece": 0.0366,
          "tau": 0.7563,
          "wasserstein": 5.8291,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9455,
          "recall": 0.875,
          "precision": 0.8556,
          "f1": 0.863,
          "bal_acc": 0.9291,
          "ece": 0.0319,
          "tau": 0.0,
          "wasserstein": 5.8291,
          "ks_pass": 100.0
        }
      },
      "SHAP-Weighted MASV-SMOTE": {
        "logistic_regression": {
          "pr_auc": 0.9668,
          "recall": 0.925,
          "precision": 0.9333,
          "f1": 0.9244,
          "bal_acc": 0.9583,
          "ece": 0.0463,
          "tau": 0.3995,
          "wasserstein": 5.885,
          "ks_pass": 100.0
        },
        "random_forest": {
          "pr_auc": 0.9466,
          "recall": 0.825,
          "precision": 0.9214,
          "f1": 0.8645,
          "bal_acc": 0.9083,
          "ece": 0.0351,
          "tau": 0.7517,
          "wasserstein": 5.885,
          "ks_pass": 100.0
        },
        "xgboost": {
          "pr_auc": 0.9485,
          "recall": 0.875,
          "precision": 0.877,
          "f1": 0.873,
          "bal_acc": 0.9305,
          "ece": 0.0295,
          "tau": 0.0,
          "wasserstein": 5.885,
          "ks_pass": 100.0
        }
      }
    }
  }
};
