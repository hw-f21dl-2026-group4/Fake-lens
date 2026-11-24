import json

#stores previously recorded model evaluation results in one place
VERIFIED_EVALUATION_DATABASE = {
    #general information about the validation
    "benchmark_info": {
        "name": "DeepFakeFace V5 Official Validation Set",
        "sample_count": 7000,
        "class_breakdown": {
            "binary": {"REAL": 3000, "FAKE": 4000},
            "multiclass": {"wiki": 3000, "text2img": 1333, "inpainting": 1333, "insight": 1334}
        },
        "description": "Cross-version standardized evaluation performed across all 16 model/task configurations on identical unseen validation portraits."
    },
    # evaluation result collected from the earlier v4 model
    "v4_models": {
        "binary": {
            "CNN": {
                "family": "CNN",
                "model_name": "V4 Dedicated Binary CNN (Phase 5H Fine-Tuned)",
                "checkpoint": "results/phase5/models/cnn_phase5h_finetune_best.pt",
                "accuracy": 0.8974,
                "balanced_accuracy": 0.8927,
                "precision": 0.8981,
                "recall": 0.9255,
                "f1_score": 0.9116,
                "roc_auc": 0.9677,
                "real_recall": 0.8600,
                "fake_recall": 0.9255,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2580, 420], [298, 3702]]
                },
                "is_champion": True
            },
            "MLP": {
                "family": "MLP",
                "model_name": "V4 Binary MLP (Phase 5H Optimized)",
                "checkpoint": "results/phase5/phase5h_classical_v4/models/mlp_phase5h_v4_best.joblib",
                "accuracy": 0.7376,
                "balanced_accuracy": 0.7406,
                "precision": 0.8012,
                "recall": 0.7193,
                "f1_score": 0.7580,
                "roc_auc": 0.8277,
                "real_recall": 0.7620,
                "fake_recall": 0.7193,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2286, 714], [1123, 2877]]
                },
                "is_champion": False
            },
            "Random Forest": {
                "family": "Random Forest",
                "model_name": "V4 Binary Random Forest (Phase 5H Optimized)",
                "checkpoint": "results/phase5/phase5h_classical_v4/models/random_forest_phase5h_v4_best.joblib",
                "accuracy": 0.7044,
                "balanced_accuracy": 0.7033,
                "precision": 0.7569,
                "recall": 0.7113,
                "f1_score": 0.7333,
                "roc_auc": 0.7813,
                "real_recall": 0.6953,
                "fake_recall": 0.7113,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2086, 914], [1155, 2845]]
                },
                "is_champion": False
            },
            "KNN": {
                "family": "KNN",
                "model_name": "V4 Binary KNN (Phase 5H Optimized)",
                "checkpoint": "results/phase5/phase5h_classical_v4/models/knn_phase5h_v4_best.joblib",
                "accuracy": 0.6077,
                "balanced_accuracy": 0.6178,
                "precision": 0.7008,
                "recall": 0.5470,
                "f1_score": 0.6144,
                "roc_auc": 0.6702,
                "real_recall": 0.6887,
                "fake_recall": 0.5470,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2066, 934], [1812, 2188]]
                },
                "is_champion": False
            }
        },
        #result for model that identity the image generation category
        "multiclass": {
            "CNN": {
                "family": "CNN",
                "model_name": "V4 Multiclass CNN (Phase 6C Continuation)",
                "checkpoint": "results/phase6/cnn_continuation/models/cnn_phase6c_best.pt",
                "accuracy": 0.8243,
                "macro_f1": 0.7931,
                "weighted_f1": 0.8252,
                "balanced_accuracy": 0.7995,
                "wiki_recall": 0.9037,
                "text2img_recall": 0.9197,
                "inpainting_recall": 0.8087,
                "insight_recall": 0.5660,
                "wiki_precision": 0.9439,
                "text2img_precision": 0.9015,
                "inpainting_precision": 0.6525,
                "insight_precision": 0.6771,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2711, 14, 62, 213],
                        [1, 1226, 103, 3],
                        [20, 91, 1078, 144],
                        [141, 29, 409, 755]
                    ]
                },
                "is_champion": True
            },
            "MLP": {
                "family": "MLP",
                "model_name": "V4 Multiclass MLP (Phase 6 Optimized)",
                "checkpoint": "results/phase6/classical_optimization/models/mlp_phase6_optimized_best.joblib",
                "accuracy": 0.7341,
                "macro_f1": 0.6904,
                "weighted_f1": 0.7360,
                "balanced_accuracy": 0.6864,
                "wiki_recall": 0.8870,
                "text2img_recall": 0.8755,
                "inpainting_recall": 0.6002,
                "insight_recall": 0.3831,
                "wiki_precision": 0.7651,
                "text2img_precision": 0.8537,
                "inpainting_precision": 0.6375,
                "insight_precision": 0.5678,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2661, 55, 109, 175],
                        [82, 1167, 70, 14],
                        [220, 113, 800, 200],
                        [515, 32, 276, 511]
                    ]
                },
                "is_champion": False
            },
            "Random Forest": {
                "family": "Random Forest",
                "model_name": "V4 Multiclass Random Forest (Phase 6E Final Refit)",
                "checkpoint": "results/phase6/random_forest_phase6e/models/random_forest_phase6e_final_refit.joblib",
                "accuracy": 0.7003,
                "macro_f1": 0.6607,
                "weighted_f1": 0.7001,
                "balanced_accuracy": 0.6641,
                "wiki_recall": 0.8163,
                "text2img_recall": 0.8425,
                "inpainting_recall": 0.5679,
                "insight_recall": 0.4295,
                "wiki_precision": 0.8112,
                "text2img_precision": 0.7517,
                "inpainting_precision": 0.5620,
                "insight_precision": 0.5426,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2449, 131, 153, 267],
                        [89, 1123, 103, 18],
                        [204, 174, 757, 198],
                        [377, 66, 318, 573]
                    ]
                },
                "is_champion": False
            },
            "KNN": {
                "family": "KNN",
                "model_name": "V4 Multiclass KNN (Phase 6 Best)",
                "checkpoint": "results/phase6/classical/models/knn_phase6_multiclass_best.joblib",
                "accuracy": 0.5041,
                "macro_f1": 0.3687,
                "weighted_f1": 0.4497,
                "balanced_accuracy": 0.4094,
                "wiki_recall": 0.8073,
                "text2img_recall": 0.7682,
                "inpainting_recall": 0.0293,
                "insight_recall": 0.0330,
                "wiki_precision": 0.4896,
                "text2img_precision": 0.7642,
                "inpainting_precision": 0.3578,
                "insight_precision": 0.1760,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2422, 154, 40, 384],
                        [289, 1024, 9, 11],
                        [1014, 116, 39, 164],
                        [1222, 47, 21, 44]
                    ]
                },
                "is_champion": False
            }
        }
    },
    "v5_models": {
        "binary": {
            "CNN": {
                "family": "CNN",
                "model_name": "V5 Binary CNN (Phase 7E.5/7F)",
                "checkpoint": "results/phase7/binary_cnn/epoch5_continuation/models/cnn_phase7e1_best.pt",
                "accuracy": 0.7591,
                "balanced_accuracy": 0.7410,
                "precision": 0.7500,
                "recall": 0.8680,
                "f1_score": 0.8046,
                "roc_auc": 0.8638,
                "real_recall": 0.6140,
                "fake_recall": 0.8680,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[1842, 1158], [528, 3472]]
                },
                "is_champion": False
            },
            "MLP": {
                "family": "MLP",
                "model_name": "V5 Binary MLP (Phase 7C)",
                "checkpoint": "results/phase7/binary_mlp/models/mlp_v5_binary_best.joblib",
                "accuracy": 0.8426,
                "balanced_accuracy": 0.8415,
                "precision": 0.8723,
                "recall": 0.8488,
                "f1_score": 0.8604,
                "roc_auc": 0.9229,
                "real_recall": 0.8343,
                "fake_recall": 0.8488,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2503, 497], [605, 3395]]
                },
                "is_champion": False
            },
            "Random Forest": {
                "family": "Random Forest",
                "model_name": "V5 Binary Random Forest (Phase 7B)",
                "checkpoint": "results/phase7/binary_rf/models/random_forest_v5_binary_best.joblib",
                "accuracy": 0.8484,
                "balanced_accuracy": 0.8438,
                "precision": 0.8610,
                "recall": 0.8763,
                "f1_score": 0.8685,
                "roc_auc": 0.9267,
                "real_recall": 0.8113,
                "fake_recall": 0.8763,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[2434, 566], [495, 3505]]
                },
                "is_champion": False
            },
            "KNN": {
                "family": "KNN",
                "model_name": "V5 Binary KNN (Phase 7D)",
                "checkpoint": "results/phase7/binary_knn/models/knn_v5_binary_best.joblib",
                "accuracy": 0.6136,
                "balanced_accuracy": 0.6046,
                "precision": 0.6370,
                "recall": 0.7525,
                "f1_score": 0.6899,
                "roc_auc": 0.6837,
                "real_recall": 0.4283,
                "fake_recall": 0.7525,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["REAL", "FAKE"],
                    "matrix": [[1285, 1715], [990, 3010]]
                },
                "is_champion": False
            }
        },
        "multiclass": {
            "CNN": {
                "family": "CNN",
                "model_name": "V5 Multiclass CNN (Phase 9D Ep25)",
                "checkpoint": "results/phase9/v5_multiclass_cnn_architecture/phase9d_epoch21_to25/models/cnn_v5_phase9d_best_holdout.pt",
                "accuracy": 0.7294,
                "macro_f1": 0.7099,
                "weighted_f1": 0.7320,
                "balanced_accuracy": 0.7125,
                "wiki_recall": 0.7837,
                "text2img_recall": 0.9430,
                "inpainting_recall": 0.7314,
                "insight_recall": 0.3921,
                "wiki_precision": 0.7405,
                "text2img_precision": 0.9155,
                "inpainting_precision": 0.6766,
                "insight_precision": 0.5173,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2351, 22, 254, 373],
                        [9, 1257, 61, 6],
                        [168, 81, 975, 109],
                        [647, 13, 151, 523]
                    ]
                },
                "is_champion": False
            },
            "MLP": {
                "family": "MLP",
                "model_name": "V5 Multiclass MLP (Phase 8B)",
                "checkpoint": "results/phase8/v5_multiclass_mlp/models/mlp_v5_multiclass_best.joblib",
                "accuracy": 0.7369,
                "macro_f1": 0.7132,
                "weighted_f1": 0.7371,
                "balanced_accuracy": 0.7111,
                "wiki_recall": 0.8193,
                "text2img_recall": 0.8590,
                "inpainting_recall": 0.6062,
                "insight_recall": 0.5600,
                "wiki_precision": 0.8301,
                "text2img_precision": 0.8814,
                "inpainting_precision": 0.6972,
                "insight_precision": 0.4838,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2458, 57, 70, 415],
                        [55, 1145, 93, 40],
                        [104, 79, 808, 342],
                        [345, 18, 224, 747]
                    ]
                },
                "is_champion": False
            },
            "Random Forest": {
                "family": "Random Forest",
                "model_name": "V5 Multiclass Random Forest (Phase 8A)",
                "checkpoint": "results/phase8/v5_multiclass_rf/models/random_forest_v5_multiclass_best.joblib",
                "accuracy": 0.6973,
                "macro_f1": 0.6629,
                "weighted_f1": 0.7020,
                "balanced_accuracy": 0.6702,
                "wiki_recall": 0.7840,
                "text2img_recall": 0.8515,
                "inpainting_recall": 0.5904,
                "insight_recall": 0.4550,
                "wiki_precision": 0.8141,
                "text2img_precision": 0.7492,
                "inpainting_precision": 0.5601,
                "insight_precision": 0.5105,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2352, 145, 153, 350],
                        [75, 1135, 102, 21],
                        [163, 170, 787, 213],
                        [299, 65, 363, 607]
                    ]
                },
                "is_champion": False
            },
            "KNN": {
                "family": "KNN",
                "model_name": "V5 Multiclass KNN (Phase 8C)",
                "checkpoint": "results/phase8/v5_multiclass_knn/models/knn_v5_multiclass_best.joblib",
                "accuracy": 0.5060,
                "macro_f1": 0.4317,
                "weighted_f1": 0.4988,
                "balanced_accuracy": 0.4280,
                "wiki_recall": 0.7557,
                "text2img_recall": 0.6542,
                "inpainting_recall": 0.1845,
                "insight_recall": 0.1177,
                "wiki_precision": 0.5141,
                "text2img_precision": 0.8052,
                "inpainting_precision": 0.3942,
                "insight_precision": 0.1778,
                "support": 7000,
                "confusion_matrix": {
                    "labels": ["wiki", "text2img", "inpainting", "insight"],
                    "matrix": [
                        [2267, 95, 179, 459],
                        [330, 872, 80, 51],
                        [781, 90, 246, 216],
                        [1032, 26, 119, 157]
                    ]
                },
                "is_champion": False
            }
        }
    }
}
