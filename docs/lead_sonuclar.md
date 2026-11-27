# LEAD1.0 sonuçları — val ve FİNAL TEST (2026-11-26)

Test **bir kez**, kullanıcının açık onayıyla çalıştırıldı (`python -m src.pipelines.lead_run --stage test --final-degerlendirme`). Testte hiçbir şey seçilmedi: ayarlar, öznitelik grupları, eşikler (val F1-max), R3'ün k'sı ve L5-FE ağırlıkları val kayıtlarından aynen alındı (protokol §21.5). Sonuçlara göre hiçbir şey değiştirilmedi. Bina bölmesi, 3 seed (42, 7, 2026), her seed'de 40 test binası. Değerler 3 seed ort. ± std. Bütün sayılar `results/lead/tables/` dosyalarından betikle (`scripts/lead_results_report.py`) alınmıştır.

**Final model (val'de §18.2 + §21 ile seçildi): L1 / behavior.** Birincil metrik `other` (davranışsal) dilimi PR-AUC; genel ROC-AUC/PR-AUC literatür kıyası için.

## 1. Ana tablo: `other` ve genel, val ile test yan yana

| model | set | other PR-AUC val | other PR-AUC test | genel PR-AUC val | genel PR-AUC test | genel ROC-AUC val | genel ROC-AUC test | other F1 test | other olay recall test |
|---|---|---|---|---|---|---|---|---|---|
| L1 **(final)** | behavior | 0.058 ± 0.011 | 0.051 ± 0.029 | 0.575 ± 0.017 | 0.646 ± 0.021 | 0.857 ± 0.005 | 0.899 ± 0.008 | 0.101 ± 0.039 | 0.203 ± 0.071 |
| L1-FE | full | 0.053 ± 0.030 | 0.120 ± 0.009 | 0.667 ± 0.099 | 0.819 ± 0.023 | 0.912 ± 0.033 | 0.954 ± 0.006 | 0.105 ± 0.091 | 0.253 ± 0.173 |
| L1-FE | behavior | 0.055 ± 0.039 | 0.078 ± 0.049 | 0.572 ± 0.035 | 0.686 ± 0.078 | 0.863 ± 0.016 | 0.915 ± 0.030 | 0.093 ± 0.087 | 0.193 ± 0.198 |
| L0-R3 | - | 0.007 ± 0.002 | 0.005 ± 0.001 | 0.225 ± 0.154 | 0.398 ± 0.065 | 0.726 ± 0.051 | 0.772 ± 0.009 | 0.000 ± 0.000 | 0.108 ± 0.039 |
| L0-A0 | - | 0.057 ± 0.006 | 0.035 ± 0.017 | 0.361 ± 0.011 | 0.383 ± 0.029 | 0.720 ± 0.028 | 0.735 ± 0.020 | 0.110 ± 0.051 | 0.367 ± 0.084 |
| L4 | - | 0.055 ± 0.022 | 0.094 ± 0.014 | 0.140 ± 0.039 | 0.158 ± 0.004 | 0.601 ± 0.013 | 0.593 ± 0.021 | 0.176 ± 0.027 | 0.587 ± 0.063 |
| L4-semi | - | 0.048 ± 0.015 | 0.075 ± 0.030 | 0.431 ± 0.035 | 0.490 ± 0.038 | 0.718 ± 0.023 | 0.727 ± 0.031 | 0.143 ± 0.046 | 0.324 ± 0.079 |
| L2-FE | full | 0.045 ± 0.019 | 0.109 ± 0.035 | 0.594 ± 0.086 | 0.768 ± 0.034 | 0.903 ± 0.037 | 0.939 ± 0.007 | 0.134 ± 0.091 | 0.224 ± 0.091 |
| L2-FE | behavior | 0.075 ± 0.062 | 0.106 ± 0.029 | 0.516 ± 0.086 | 0.650 ± 0.100 | 0.852 ± 0.015 | 0.895 ± 0.040 | 0.155 ± 0.087 | 0.257 ± 0.091 |
| L5-FE | full | 0.053 ± 0.031 | 0.125 ± 0.017 | 0.668 ± 0.101 | 0.818 ± 0.025 | 0.914 ± 0.036 | 0.956 ± 0.004 | 0.111 ± 0.101 | 0.264 ± 0.191 |
| L5-FE | behavior | 0.074 ± 0.057 | 0.108 ± 0.042 | 0.583 ± 0.038 | 0.692 ± 0.084 | 0.872 ± 0.015 | 0.923 ± 0.028 | 0.131 ± 0.112 | 0.245 ± 0.219 |
| IF | behavior | 0.012 ± 0.003 | 0.008 ± 0.004 | 0.089 ± 0.051 | 0.200 ± 0.185 | 0.752 ± 0.023 | 0.762 ± 0.030 | 0.017 ± 0.015 | 0.083 ± 0.081 |
| L1-OFF | full | 0.057 ± 0.032 | 0.123 ± 0.020 | 0.707 ± 0.109 | 0.867 ± 0.034 | 0.938 ± 0.035 | 0.972 ± 0.006 | 0.103 ± 0.048 | 0.288 ± 0.089 |

PR-AUC tabanı (other): val 0.0073, test 0.0050.

PR-AUC tabanı (genel): val 0.0230, test 0.0228.

## 2. Bütün dilimler ve metrikler (TEST)

Taban = dilimin pozitif oranı (rastgele skorun beklenen PR-AUC'si). Val karşılıkları: `results/lead/tables/test_summary_val_vs_test.csv` (kume = val).

### other (PR-AUC tabanı ≈ 0.0050)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.774 ± 0.014 | 0.051 ± 0.029 | 0.146 ± 0.074 | 0.079 ± 0.022 | 0.101 ± 0.039 | 0.203 ± 0.071 |
| L1-FE | full | 0.838 ± 0.010 | 0.120 ± 0.009 | 0.306 ± 0.139 | 0.074 ± 0.077 | 0.105 ± 0.091 | 0.253 ± 0.173 |
| L1-FE | behavior | 0.814 ± 0.034 | 0.078 ± 0.049 | 0.169 ± 0.142 | 0.065 ± 0.060 | 0.093 ± 0.087 | 0.193 ± 0.198 |
| L0-R3 | - | 0.497 ± 0.001 | 0.005 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.108 ± 0.039 |
| L0-A0 | - | 0.630 ± 0.016 | 0.035 ± 0.017 | 0.112 ± 0.050 | 0.109 ± 0.053 | 0.110 ± 0.051 | 0.367 ± 0.084 |
| L4 | - | 0.712 ± 0.052 | 0.094 ± 0.014 | 0.157 ± 0.015 | 0.205 ± 0.054 | 0.176 ± 0.027 | 0.587 ± 0.063 |
| L4-semi | - | 0.704 ± 0.058 | 0.075 ± 0.030 | 0.296 ± 0.055 | 0.095 ± 0.034 | 0.143 ± 0.046 | 0.324 ± 0.079 |
| L2-FE | full | 0.832 ± 0.016 | 0.109 ± 0.035 | 0.269 ± 0.109 | 0.095 ± 0.071 | 0.134 ± 0.091 | 0.224 ± 0.091 |
| L2-FE | behavior | 0.830 ± 0.014 | 0.106 ± 0.029 | 0.299 ± 0.009 | 0.114 ± 0.073 | 0.155 ± 0.087 | 0.257 ± 0.091 |
| L5-FE | full | 0.844 ± 0.017 | 0.125 ± 0.017 | 0.300 ± 0.141 | 0.085 ± 0.095 | 0.111 ± 0.101 | 0.264 ± 0.191 |
| L5-FE | behavior | 0.846 ± 0.026 | 0.108 ± 0.042 | 0.209 ± 0.123 | 0.101 ± 0.089 | 0.131 ± 0.112 | 0.245 ± 0.219 |
| IF | behavior | 0.614 ± 0.044 | 0.008 ± 0.004 | 0.010 ± 0.007 | 0.108 ± 0.150 | 0.017 ± 0.015 | 0.083 ± 0.081 |
| L1-OFF | full | 0.878 ± 0.011 | 0.123 ± 0.020 | 0.279 ± 0.069 | 0.067 ± 0.036 | 0.103 ± 0.048 | 0.288 ± 0.089 |

### genel (PR-AUC tabanı ≈ 0.0228)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.899 ± 0.008 | 0.646 ± 0.021 | 0.834 ± 0.035 | 0.543 ± 0.051 | 0.656 ± 0.026 | 0.312 ± 0.001 |
| L1-FE | full | 0.954 ± 0.006 | 0.819 ± 0.023 | 0.944 ± 0.067 | 0.645 ± 0.052 | 0.764 ± 0.014 | 0.525 ± 0.092 |
| L1-FE | behavior | 0.915 ± 0.030 | 0.686 ± 0.078 | 0.911 ± 0.064 | 0.522 ± 0.023 | 0.663 ± 0.036 | 0.295 ± 0.066 |
| L0-R3 | - | 0.772 ± 0.009 | 0.398 ± 0.065 | 0.707 ± 0.132 | 0.550 ± 0.021 | 0.614 ± 0.045 | 0.556 ± 0.041 |
| L0-A0 | - | 0.735 ± 0.020 | 0.383 ± 0.029 | 0.647 ± 0.038 | 0.331 ± 0.029 | 0.437 ± 0.034 | 0.253 ± 0.042 |
| L4 | - | 0.593 ± 0.021 | 0.158 ± 0.004 | 0.398 ± 0.011 | 0.150 ± 0.009 | 0.218 ± 0.010 | 0.397 ± 0.039 |
| L4-semi | - | 0.727 ± 0.031 | 0.490 ± 0.038 | 0.899 ± 0.027 | 0.418 ± 0.029 | 0.570 ± 0.032 | 0.297 ± 0.019 |
| L2-FE | full | 0.939 ± 0.007 | 0.768 ± 0.034 | 0.922 ± 0.066 | 0.597 ± 0.035 | 0.722 ± 0.015 | 0.341 ± 0.050 |
| L2-FE | behavior | 0.895 ± 0.040 | 0.650 ± 0.100 | 0.893 ± 0.076 | 0.498 ± 0.049 | 0.639 ± 0.060 | 0.296 ± 0.018 |
| L5-FE | full | 0.956 ± 0.004 | 0.818 ± 0.025 | 0.931 ± 0.090 | 0.641 ± 0.045 | 0.756 ± 0.009 | 0.523 ± 0.089 |
| L5-FE | behavior | 0.923 ± 0.028 | 0.692 ± 0.084 | 0.897 ± 0.092 | 0.520 ± 0.020 | 0.658 ± 0.041 | 0.301 ± 0.067 |
| IF | behavior | 0.762 ± 0.030 | 0.200 ± 0.185 | 0.288 ± 0.216 | 0.438 ± 0.033 | 0.311 ± 0.144 | 0.233 ± 0.019 |
| L1-OFF | full | 0.972 ± 0.006 | 0.867 ± 0.034 | 0.945 ± 0.048 | 0.711 ± 0.025 | 0.811 ± 0.012 | 0.719 ± 0.123 |

### gap_edge (PR-AUC tabanı ≈ 0.0030)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.810 ± 0.017 | 0.252 ± 0.010 | 0.252 ± 0.033 | 0.271 ± 0.022 | 0.259 ± 0.007 | 0.096 ± 0.013 |
| L1-FE | full | 0.942 ± 0.024 | 0.623 ± 0.043 | 0.716 ± 0.252 | 0.548 ± 0.092 | 0.591 ± 0.070 | 0.451 ± 0.116 |
| L1-FE | behavior | 0.826 ± 0.073 | 0.292 ± 0.069 | 0.470 ± 0.280 | 0.248 ± 0.052 | 0.301 ± 0.048 | 0.079 ± 0.061 |
| L0-R3 | - | 0.795 ± 0.004 | 0.166 ± 0.068 | 0.278 ± 0.120 | 0.596 ± 0.011 | 0.368 ± 0.105 | 0.528 ± 0.015 |
| L0-A0 | - | 0.554 ± 0.014 | 0.100 ± 0.012 | 0.099 ± 0.004 | 0.150 ± 0.007 | 0.119 ± 0.001 | 0.057 ± 0.006 |
| L4 | - | 0.744 ± 0.018 | 0.175 ± 0.020 | 0.140 ± 0.023 | 0.279 ± 0.018 | 0.186 ± 0.024 | 0.197 ± 0.037 |
| L4-semi | - | 0.740 ± 0.039 | 0.245 ± 0.015 | 0.436 ± 0.089 | 0.266 ± 0.006 | 0.329 ± 0.030 | 0.120 ± 0.014 |
| L2-FE | full | 0.881 ± 0.029 | 0.379 ± 0.037 | 0.520 ± 0.207 | 0.333 ± 0.070 | 0.381 ± 0.027 | 0.177 ± 0.082 |
| L2-FE | behavior | 0.847 ± 0.050 | 0.256 ± 0.092 | 0.436 ± 0.237 | 0.259 ± 0.040 | 0.303 ± 0.026 | 0.102 ± 0.030 |
| L5-FE | full | 0.939 ± 0.025 | 0.606 ± 0.063 | 0.694 ± 0.290 | 0.549 ± 0.095 | 0.574 ± 0.098 | 0.451 ± 0.115 |
| L5-FE | behavior | 0.847 ± 0.062 | 0.313 ± 0.080 | 0.484 ± 0.341 | 0.259 ± 0.063 | 0.299 ± 0.031 | 0.093 ± 0.072 |
| IF | behavior | 0.602 ± 0.037 | 0.042 ± 0.054 | 0.030 ± 0.027 | 0.220 ± 0.029 | 0.049 ± 0.039 | 0.073 ± 0.040 |
| L1-OFF | full | 0.999 ± 0.000 | 0.849 ± 0.040 | 0.749 ± 0.154 | 0.811 ± 0.154 | 0.758 ± 0.039 | 0.765 ± 0.190 |

### frozen (PR-AUC tabanı ≈ 0.0151)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.958 ± 0.017 | 0.813 ± 0.038 | 0.820 ± 0.037 | 0.749 ± 0.077 | 0.780 ± 0.026 | 0.947 ± 0.018 |
| L1-FE | full | 0.994 ± 0.001 | 0.958 ± 0.016 | 0.934 ± 0.080 | 0.852 ± 0.090 | 0.886 ± 0.015 | 0.928 ± 0.048 |
| L1-FE | behavior | 0.964 ± 0.024 | 0.839 ± 0.089 | 0.902 ± 0.071 | 0.723 ± 0.049 | 0.803 ± 0.057 | 0.916 ± 0.039 |
| L0-R3 | - | 0.857 ± 0.015 | 0.493 ± 0.121 | 0.675 ± 0.144 | 0.719 ± 0.026 | 0.693 ± 0.090 | 0.995 ± 0.003 |
| L0-A0 | - | 0.805 ± 0.024 | 0.470 ± 0.014 | 0.615 ± 0.038 | 0.439 ± 0.012 | 0.513 ± 0.021 | 0.657 ± 0.055 |
| L4 | - | 0.527 ± 0.031 | 0.085 ± 0.004 | 0.238 ± 0.010 | 0.108 ± 0.002 | 0.148 ± 0.003 | 0.757 ± 0.045 |
| L4-semi | - | 0.733 ± 0.032 | 0.574 ± 0.020 | 0.885 ± 0.032 | 0.552 ± 0.013 | 0.680 ± 0.020 | 0.723 ± 0.039 |
| L2-FE | full | 0.984 ± 0.006 | 0.908 ± 0.016 | 0.912 ± 0.076 | 0.813 ± 0.056 | 0.856 ± 0.005 | 0.851 ± 0.025 |
| L2-FE | behavior | 0.924 ± 0.046 | 0.767 ± 0.119 | 0.878 ± 0.088 | 0.669 ± 0.088 | 0.759 ± 0.088 | 0.813 ± 0.046 |
| L5-FE | full | 0.994 ± 0.001 | 0.953 ± 0.013 | 0.919 ± 0.107 | 0.840 ± 0.070 | 0.872 ± 0.021 | 0.911 ± 0.033 |
| L5-FE | behavior | 0.960 ± 0.026 | 0.826 ± 0.096 | 0.883 ± 0.107 | 0.705 ± 0.050 | 0.783 ± 0.071 | 0.859 ± 0.048 |
| IF | behavior | 0.839 ± 0.045 | 0.238 ± 0.246 | 0.270 ± 0.218 | 0.583 ± 0.036 | 0.331 ± 0.192 | 0.757 ± 0.019 |
| L1-OFF | full | 0.997 ± 0.001 | 0.971 ± 0.010 | 0.934 ± 0.059 | 0.901 ± 0.051 | 0.915 ± 0.004 | 0.944 ± 0.055 |

### paylasilan (PR-AUC tabanı ≈ 0.0163)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.917 ± 0.010 | 0.669 ± 0.012 | 0.795 ± 0.041 | 0.586 ± 0.047 | 0.672 ± 0.018 | 0.306 ± 0.005 |
| L1-FE | full | 0.974 ± 0.006 | 0.865 ± 0.009 | 0.930 ± 0.084 | 0.719 ± 0.075 | 0.806 ± 0.015 | 0.549 ± 0.087 |
| L1-FE | behavior | 0.930 ± 0.033 | 0.715 ± 0.093 | 0.888 ± 0.081 | 0.568 ± 0.036 | 0.692 ± 0.048 | 0.285 ± 0.065 |
| L0-R3 | - | 0.809 ± 0.004 | 0.421 ± 0.091 | 0.664 ± 0.140 | 0.623 ± 0.005 | 0.639 ± 0.069 | 0.567 ± 0.021 |
| L0-A0 | - | 0.737 ± 0.020 | 0.390 ± 0.015 | 0.587 ± 0.036 | 0.360 ± 0.015 | 0.446 ± 0.022 | 0.235 ± 0.031 |
| L4 | - | 0.584 ± 0.019 | 0.144 ± 0.003 | 0.331 ± 0.011 | 0.157 ± 0.002 | 0.213 ± 0.003 | 0.394 ± 0.037 |
| L4-semi | - | 0.734 ± 0.024 | 0.518 ± 0.018 | 0.877 ± 0.032 | 0.468 ± 0.018 | 0.610 ± 0.019 | 0.310 ± 0.021 |
| L2-FE | full | 0.957 ± 0.006 | 0.816 ± 0.017 | 0.905 ± 0.080 | 0.679 ± 0.060 | 0.772 ± 0.018 | 0.358 ± 0.061 |
| L2-FE | behavior | 0.910 ± 0.042 | 0.687 ± 0.106 | 0.869 ± 0.092 | 0.554 ± 0.048 | 0.676 ± 0.062 | 0.306 ± 0.027 |
| L5-FE | full | 0.974 ± 0.006 | 0.863 ± 0.012 | 0.916 ± 0.109 | 0.717 ± 0.073 | 0.798 ± 0.011 | 0.549 ± 0.087 |
| L5-FE | behavior | 0.935 ± 0.030 | 0.724 ± 0.096 | 0.874 ± 0.111 | 0.575 ± 0.025 | 0.692 ± 0.040 | 0.301 ± 0.072 |
| IF | behavior | 0.760 ± 0.031 | 0.196 ± 0.203 | 0.248 ± 0.204 | 0.470 ± 0.021 | 0.289 ± 0.166 | 0.235 ± 0.023 |
| L1-OFF | full | 0.988 ± 0.004 | 0.916 ± 0.016 | 0.932 ± 0.060 | 0.798 ± 0.048 | 0.858 ± 0.007 | 0.777 ± 0.137 |

### tekil (PR-AUC tabanı ≈ 0.0067)

| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |
|---|---|---|---|---|---|---|---|
| L1 | behavior | 0.857 ± 0.012 | 0.474 ± 0.068 | 0.542 ± 0.066 | 0.444 ± 0.085 | 0.482 ± 0.043 | 0.332 ± 0.024 |
| L1-FE | full | 0.907 ± 0.018 | 0.591 ± 0.063 | 0.811 ± 0.185 | 0.467 ± 0.024 | 0.585 ± 0.038 | 0.457 ± 0.117 |
| L1-FE | behavior | 0.879 ± 0.021 | 0.484 ± 0.052 | 0.726 ± 0.176 | 0.410 ± 0.044 | 0.518 ± 0.056 | 0.322 ± 0.065 |
| L0-R3 | - | 0.682 ± 0.024 | 0.127 ± 0.046 | 0.341 ± 0.154 | 0.370 ± 0.050 | 0.337 ± 0.073 | 0.530 ± 0.092 |
| L0-A0 | - | 0.730 ± 0.033 | 0.214 ± 0.047 | 0.292 ± 0.053 | 0.261 ± 0.057 | 0.275 ± 0.054 | 0.300 ± 0.069 |
| L4 | - | 0.617 ± 0.031 | 0.065 ± 0.005 | 0.143 ± 0.004 | 0.133 ± 0.022 | 0.137 ± 0.014 | 0.404 ± 0.045 |
| L4-semi | - | 0.711 ± 0.051 | 0.301 ± 0.052 | 0.646 ± 0.073 | 0.297 ± 0.055 | 0.406 ± 0.067 | 0.261 ± 0.023 |
| L2-FE | full | 0.893 ± 0.017 | 0.489 ± 0.046 | 0.719 ± 0.172 | 0.396 ± 0.007 | 0.506 ± 0.047 | 0.299 ± 0.027 |
| L2-FE | behavior | 0.857 ± 0.030 | 0.411 ± 0.085 | 0.667 ± 0.195 | 0.362 ± 0.048 | 0.467 ± 0.082 | 0.269 ± 0.006 |
| L5-FE | full | 0.911 ± 0.016 | 0.580 ± 0.078 | 0.783 ± 0.234 | 0.457 ± 0.007 | 0.567 ± 0.069 | 0.451 ± 0.107 |
| L5-FE | behavior | 0.892 ± 0.018 | 0.467 ± 0.084 | 0.694 ± 0.250 | 0.388 ± 0.063 | 0.492 ± 0.101 | 0.303 ± 0.053 |
| IF | behavior | 0.767 ± 0.034 | 0.067 ± 0.075 | 0.090 ± 0.075 | 0.357 ± 0.072 | 0.127 ± 0.079 | 0.229 ± 0.022 |
| L1-OFF | full | 0.933 ± 0.017 | 0.617 ± 0.081 | 0.794 ± 0.141 | 0.500 ± 0.019 | 0.611 ± 0.053 | 0.558 ± 0.107 |

## 3. Bina bazında F1 (TEST; anomalisi olan test binaları, 3 seed birleşik)

| model | set | bina sayısı | medyan F1 | çeyrekler arası (Q1–Q3) |
|---|---|---|---|---|
| L1 | behavior | 120 | 0.720 | 0.257–0.812 |
| L1-FE | full | 120 | 0.832 | 0.629–0.902 |
| L1-FE | behavior | 120 | 0.710 | 0.337–0.819 |
| L0-R3 | - | 120 | 0.650 | 0.435–0.814 |
| L0-A0 | - | 120 | 0.482 | 0.225–0.600 |
| L4 | - | 120 | 0.208 | 0.109–0.316 |
| L4-semi | - | 120 | 0.621 | 0.094–0.791 |
| L2-FE | full | 120 | 0.787 | 0.503–0.870 |
| L2-FE | behavior | 120 | 0.707 | 0.311–0.814 |
| L5-FE | full | 120 | 0.823 | 0.584–0.900 |
| L5-FE | behavior | 120 | 0.709 | 0.301–0.821 |
| IF | behavior | 120 | 0.513 | 0.049–0.711 |
| L1-OFF | full | 120 | 0.890 | 0.714–0.941 |

## 4. Val yeniden üretim kontrolü (§21.6)

Val aşamasında model nesneleri kaydedilmediği için her model val prosedürüyle yeniden eğitildi; önce val skorları yeniden üretilip kayıtlı val skorlarıyla karşılaştırıldı (`test_val_reproduction.csv`).

| model | set | seed | max |fark| (skor) | val PR-AUC kayıtlı | val PR-AUC yeniden |
|---|---|---|---|---|---|
| L0-R3 | - | 42 | 0.00e+00 | 0.1845 | 0.1845 |
| L0-A0 | - | 42 | 0.00e+00 | 0.3495 | 0.3495 |
| L1 | behavior | 42 | 0.00e+00 | 0.5619 | 0.5619 |
| L4 | - | 42 | 0.00e+00 | 0.1421 | 0.1421 |
| L4-semi | - | 42 | 0.00e+00 | 0.4309 | 0.4309 |
| L1-FE | full | 42 | 0.00e+00 | 0.6292 | 0.6292 |
| L1-OFF | full | 42 | 0.00e+00 | 0.6614 | 0.6614 |
| L1-FE | behavior | 42 | 0.00e+00 | 0.5733 | 0.5733 |
| L2-FE | full | 42 | 0.00e+00 | 0.5479 | 0.5479 |
| L2-FE | behavior | 42 | 0.00e+00 | 0.5146 | 0.5146 |
| L5-FE | full | 42 | 0.00e+00 | 0.6292 | 0.6292 |
| L5-FE | behavior | 42 | 5.96e-08 | 0.5781 | 0.5781 |
| IF | behavior | 42 | 0.00e+00 | 0.0592 | 0.0592 |
| L0-R3 | - | 7 | 0.00e+00 | 0.3953 | 0.3953 |
| L0-A0 | - | 7 | 0.00e+00 | 0.3611 | 0.3611 |
| L1 | behavior | 7 | 0.00e+00 | 0.5946 | 0.5946 |
| L4 | - | 7 | 0.00e+00 | 0.1777 | 0.1777 |
| L4-semi | - | 7 | 0.00e+00 | 0.4667 | 0.4667 |
| L1-FE | full | 7 | 0.00e+00 | 0.7786 | 0.7786 |
| L1-OFF | full | 7 | 0.00e+00 | 0.8314 | 0.8314 |
| L1-FE | behavior | 7 | 0.00e+00 | 0.6065 | 0.6065 |
| L2-FE | full | 7 | 0.00e+00 | 0.6927 | 0.6927 |
| L2-FE | behavior | 7 | 0.00e+00 | 0.6035 | 0.6035 |
| L5-FE | full | 7 | 5.96e-08 | 0.7828 | 0.7828 |
| L5-FE | behavior | 7 | 5.96e-08 | 0.6234 | 0.6234 |
| IF | behavior | 7 | 0.00e+00 | 0.1475 | 0.1475 |
| L0-R3 | - | 2026 | 0.00e+00 | 0.0966 | 0.0966 |
| L0-A0 | - | 2026 | 0.00e+00 | 0.3720 | 0.3720 |
| L1 | behavior | 2026 | 0.00e+00 | 0.5690 | 0.5690 |
| L4 | - | 2026 | 0.00e+00 | 0.1005 | 0.1005 |
| L4-semi | - | 2026 | 0.00e+00 | 0.3957 | 0.3957 |
| L1-FE | full | 2026 | 4.72e-03 | 0.5923 | 0.5923 |
| L1-OFF | full | 2026 | 0.00e+00 | 0.6284 | 0.6284 |
| L1-FE | behavior | 2026 | 0.00e+00 | 0.5374 | 0.5374 |
| L2-FE | full | 2026 | 0.00e+00 | 0.5409 | 0.5409 |
| L2-FE | behavior | 2026 | 0.00e+00 | 0.4307 | 0.4307 |
| L5-FE | full | 2026 | 4.72e-03 | 0.5923 | 0.5923 |
| L5-FE | behavior | 2026 | 5.96e-08 | 0.5477 | 0.5477 |
| IF | behavior | 2026 | 0.00e+00 | 0.0598 | 0.0598 |

## 5. Literatür kıyası (koşullar farklı; "daha iyi / daha kötü" doğrudan söylenmez, §14)

| kaynak | değer | koşul farkları |
|---|---|---|
| Kaggle LEAD birincisi (Fu, Arjunan & Miller 2022 [2]) | 0,9866 | 206 görülmemiş bina, etiketleri gizli Kaggle testi (biz: 200 etiketli binayı 120/40/40 bölüyoruz, val); nedensel kısıt yok, iki yönlü öznitelik serbest (biz: ana sonuç nedensel); private LB; yalnız ROC-AUC |
| "An Integrated Framework of Feature Engineering and Machine Learning for Large-Scale Energy Anomaly Detection", Energy Engineering (Tech Science Press), c. 123, s. 3 (yazar, yıl, sayfa doğrulanmalı) | 0,952 | bölme ve öznitelik koşulları repoda kayıtlı değil — DOĞRULANMALI; nedensellik kısıtı bilinmiyor |
| LEAD1.0 veri makalesi, PyOD yöntemleri (Gulati & Arjunan 2022 [1]) | nokta F1 ≈ 0,40–0,43 (ROC-AUC yok) | denetimsiz; 24 saatlik gün düzeyi etiket; normal/anomali günleri train/val/test'e RASTGELE dağıtılmış; bizimle kıyas yalnız Isolation Forest satırının F1'i üzerinden |
| **Bizim final (L1 / behavior, çevrimiçi)** | genel ROC-AUC 0.899 ± 0.008 (test) | nedensel öznitelikler; 200 etiketli bina 120/40/40; 3 seed |
| Bizim L1-OFF (offline, iki yönlü) | genel ROC-AUC 0.972 ± 0.006 (test) | Kaggle koşuluna en yakın satırımız; seçime girmedi |
| Bizim Isolation Forest | nokta F1 0.311 ± 0.144 (test, genel) | saat düzeyi, bina bölmesi, val'de F1-max eşik — PyOD satırıyla kıyas için |

## 6. Val → test sapmaları

Fark = test − val (3 seed ort.). Sapma, iki kümenin seed'ler arası std'lerinin büyüğünden fazlaysa "belirgin" sayıldı (ölçüt betikte; seçim için değil, yalnız yorum için).

| model | set | dilim | val PR-AUC | test PR-AUC | fark | belirgin mi |
|---|---|---|---|---|---|---|
| L1 | behavior | other | 0.058 | 0.051 | -0.007 | hayır |
| L1 | behavior | genel | 0.575 | 0.646 | +0.071 | evet |
| L1 | behavior | frozen | 0.791 | 0.813 | +0.022 | hayır |
| L1 | behavior | gap_edge | 0.292 | 0.252 | -0.041 | hayır |
| L1-FE | full | other | 0.053 | 0.120 | +0.067 | evet |
| L1-FE | full | genel | 0.667 | 0.819 | +0.153 | evet |
| L1-FE | full | frozen | 0.882 | 0.958 | +0.076 | evet |
| L1-FE | full | gap_edge | 0.505 | 0.623 | +0.118 | evet |
| L1-FE | behavior | other | 0.055 | 0.078 | +0.023 | hayır |
| L1-FE | behavior | genel | 0.572 | 0.686 | +0.113 | evet |
| L1-FE | behavior | frozen | 0.803 | 0.839 | +0.036 | hayır |
| L1-FE | behavior | gap_edge | 0.288 | 0.292 | +0.003 | hayır |
| L0-R3 | - | other | 0.007 | 0.005 | -0.002 | evet |
| L0-R3 | - | genel | 0.225 | 0.398 | +0.172 | evet |
| L0-R3 | - | frozen | 0.290 | 0.493 | +0.202 | evet |
| L0-R3 | - | gap_edge | 0.073 | 0.166 | +0.093 | evet |
| L0-A0 | - | other | 0.057 | 0.035 | -0.022 | evet |
| L0-A0 | - | genel | 0.361 | 0.383 | +0.022 | hayır |
| L0-A0 | - | frozen | 0.474 | 0.470 | -0.004 | hayır |
| L0-A0 | - | gap_edge | 0.105 | 0.100 | -0.005 | hayır |
| L4 | - | other | 0.055 | 0.094 | +0.039 | evet |
| L4 | - | genel | 0.140 | 0.158 | +0.018 | hayır |
| L4 | - | frozen | 0.076 | 0.085 | +0.009 | hayır |
| L4 | - | gap_edge | 0.169 | 0.175 | +0.006 | hayır |
| L4-semi | - | other | 0.048 | 0.075 | +0.027 | hayır |
| L4-semi | - | genel | 0.431 | 0.490 | +0.059 | evet |
| L4-semi | - | frozen | 0.560 | 0.574 | +0.014 | hayır |
| L4-semi | - | gap_edge | 0.265 | 0.245 | -0.020 | hayır |
| L2-FE | full | other | 0.045 | 0.109 | +0.064 | evet |
| L2-FE | full | genel | 0.594 | 0.768 | +0.174 | evet |
| L2-FE | full | frozen | 0.792 | 0.908 | +0.116 | evet |
| L2-FE | full | gap_edge | 0.183 | 0.379 | +0.196 | evet |
| L2-FE | behavior | other | 0.075 | 0.106 | +0.031 | hayır |
| L2-FE | behavior | genel | 0.516 | 0.650 | +0.134 | evet |
| L2-FE | behavior | frozen | 0.683 | 0.767 | +0.084 | hayır |
| L2-FE | behavior | gap_edge | 0.181 | 0.256 | +0.076 | hayır |
| L5-FE | full | other | 0.053 | 0.125 | +0.071 | evet |
| L5-FE | full | genel | 0.668 | 0.818 | +0.149 | evet |
| L5-FE | full | frozen | 0.884 | 0.953 | +0.069 | evet |
| L5-FE | full | gap_edge | 0.485 | 0.606 | +0.121 | evet |
| L5-FE | behavior | other | 0.074 | 0.108 | +0.034 | hayır |
| L5-FE | behavior | genel | 0.583 | 0.692 | +0.109 | evet |
| L5-FE | behavior | frozen | 0.797 | 0.826 | +0.029 | hayır |
| L5-FE | behavior | gap_edge | 0.294 | 0.313 | +0.020 | hayır |
| IF | behavior | other | 0.012 | 0.008 | -0.003 | hayır |
| IF | behavior | genel | 0.089 | 0.200 | +0.111 | hayır |
| IF | behavior | frozen | 0.088 | 0.238 | +0.149 | hayır |
| IF | behavior | gap_edge | 0.006 | 0.042 | +0.036 | hayır |
| L1-OFF | full | other | 0.057 | 0.123 | +0.065 | evet |
| L1-OFF | full | genel | 0.707 | 0.867 | +0.160 | evet |
| L1-OFF | full | frozen | 0.903 | 0.971 | +0.068 | evet |
| L1-OFF | full | gap_edge | 0.605 | 0.849 | +0.243 | hayır |

### 6.1 Olası nedenler: parça bileşimi (`test_split_composition.csv`; betimleyici, seçim yok)

| seed | parça | anomali_orani | other_taban | gap_edge_pay | frozen_pay | other_pay | paylasilan_olay_pay | other_paylasilan_pay | other_ilk3_bina_pay |
|---|---|---|---|---|---|---|---|---|---|
| 42 | train | 0.022 | 0.005 | 0.125 | 0.644 | 0.231 | 0.792 | 0.677 | 0.163 |
| 42 | val | 0.022 | 0.009 | 0.116 | 0.477 | 0.407 | 0.625 | 0.527 | 0.371 |
| 42 | test | 0.025 | 0.005 | 0.124 | 0.689 | 0.188 | 0.720 | 0.643 | 0.347 |
| 7 | train | 0.022 | 0.006 | 0.124 | 0.610 | 0.266 | 0.750 | 0.655 | 0.204 |
| 7 | val | 0.026 | 0.006 | 0.120 | 0.665 | 0.215 | 0.744 | 0.621 | 0.282 |
| 7 | test | 0.023 | 0.006 | 0.123 | 0.609 | 0.268 | 0.707 | 0.557 | 0.388 |
| 2026 | train | 0.024 | 0.006 | 0.124 | 0.625 | 0.251 | 0.751 | 0.641 | 0.175 |
| 2026 | val | 0.021 | 0.007 | 0.097 | 0.569 | 0.334 | 0.668 | 0.536 | 0.386 |
| 2026 | test | 0.020 | 0.004 | 0.149 | 0.669 | 0.182 | 0.769 | 0.676 | 0.319 |

Ortalama (3 seed): frozen payı val 0.570 → test 0.655; `other` payı 0.319 → 0.213; paylaşılan olay payı 0.679 → 0.732; paylaşılan `other` olay payı 0.561 → 0.625; `other` tabanı 0.0073 → 0.0050.

Yorum (kanıt değil, olası nedenler):
1. **Sapma hemen hemen bütün modellerde aynı yönde (test > val)**, kurallarda (R3) ve etiketsiz modellerde de. Bu, modelden çok **parça bileşiminden** gelen bir fark olduğunu düşündürür: test parçalarında kolay yakalanan frozen anomalilerin payı daha yüksek, `other`'ın payı daha düşük. Genel PR-AUC frozen'a çok bağlı olduğu için yükseliyor.
2. **`other` diliminde** test parçalarında paylaşılan olay payı daha yüksek; val analizinde (`lead_val_ozet.md` §5) paylaşılan olaylarda PR-AUC'nin tekil olaylardakinin yaklaşık iki katı olduğu görülmüştü.
3. **Val parçaları görece "zor" düştü** (özellikle seed 42 val: `other` payı 0,407, ilk 3 bina `other` saatlerinin %37'si). Seed başına 40 bina ve `other` saatlerinin birkaç binada yoğunlaşması, parçadan parçaya büyük oynaklık yaratıyor; val'de `other` PR-AUC'nin modelleri ayırt edememesi (§21.2 bulgusu) ile aynı nedendir.
4. **Final model (L1 behavior) ile L1-FE full arasındaki fark testte büyüdü** (`other` PR-AUC 0,051'e karşı 0,120). Val'de bu fark gürültü düzeyindeydi ve seçim kuralı basitlik sırasıyla L1 behavior'ı seçti. Protokol gereği (§13) test sonucuna göre final model **değiştirilmez**; bu, raporda "val'de ayırt edilemeyen iki modelin testte ayrışması" olarak yazılır.
5. Val skorlarında val erken durdurmasından gelen hafif iyimserlik (§8.2, val özeti) test yönündeki farkı açıklamaz (o, val'i yüksek gösterirdi); farkın yönü tersidir.

## 7. Test sonrası gözlem

> Bu bölüm test sonuçları görüldükten **sonra** yazıldı. Hiçbir seçimi, eşiği ya da ayarı değiştirmez.

**Final model değişmeden kalır: L1 / behavior** (val'de §18.2 + §21 ile seçildi; protokol §13: test sonucuna göre hiçbir şey değiştirilmez).

**L1-FE full testte daha iyi genelleşti:**

| model | other PR-AUC val | other PR-AUC test | genel PR-AUC val | genel PR-AUC test |
|---|---|---|---|---|
| L1 / behavior (final) | 0.058 ± 0.011 | 0.051 ± 0.029 | 0.575 ± 0.017 | 0.646 ± 0.021 |
| L1-FE / full | 0.053 ± 0.030 | 0.120 ± 0.009 | 0.667 ± 0.099 | 0.819 ± 0.023 |

Testte L1-FE full `other` PR-AUC'de 0.120 ± 0.009, final model 0.051 ± 0.029; genel PR-AUC'de 0.819 ve 0.646. Val'de iki model gürültü düzeyindeydi (§21.2) ve seçim kuralı basitlik sırasıyla L1 behavior'ı seçti.

**Ders:** tek bir val kümesi (seed başına 40 bina) bu büyüklükteki farkları ayırt edemedi. Seçim, aynı 40 bina üzerinde yapılan tek bir ölçüme dayandı; val parçasının bileşimi (aşağıda) sonucu modeller arasındaki farktan daha çok etkiledi. **Öneri (gelecek çalışma):** model seçimi train+val binalarında **bina bazlı (gruplu) çapraz doğrulama** ile yapılmalı (ör. 5 kat × birkaç tekrar; her katta farklı binalar tutulur); farklar kat başına eşleştirilmiş olarak ve daha fazla bağımsız ölçümle değerlendirilmeli. L-FE ablasyonu ve I-turu bu yapıyı (3 kat) zaten train içinde kullandı; final seçim de aynı yapıya taşınmalıydı.

**Bulgu: `other` skorları bina bileşimine çok duyarlı.** Aynı modelin `other` PR-AUC'si parçadan parçaya büyük ölçüde değişiyor; kurallar ve etiketsiz modeller de aynı yönde etkileniyor. Val ve test parçalarının bileşimi (`test_split_composition.csv`):

| seed | parça | bina | other tabanı | frozen payı | other payı | paylaşılan olay payı | paylaşılan other olay payı | other'ın ilk 3 binadaki payı |
|---|---|---|---|---|---|---|---|---|
| 42 | val | 40 | 0.0092 | 0.477 | 0.407 | 0.625 | 0.527 | 0.371 |
| 42 | test | 40 | 0.0049 | 0.689 | 0.188 | 0.720 | 0.643 | 0.347 |
| 7 | val | 40 | 0.0056 | 0.665 | 0.215 | 0.744 | 0.621 | 0.282 |
| 7 | test | 40 | 0.0064 | 0.609 | 0.268 | 0.707 | 0.557 | 0.388 |
| 2026 | val | 40 | 0.0072 | 0.569 | 0.334 | 0.668 | 0.536 | 0.386 |
| 2026 | test | 40 | 0.0036 | 0.669 | 0.182 | 0.769 | 0.676 | 0.319 |

`other` saatlerinin %28–39'u parçanın yalnız 3 binasında; bu yüzden 40 binalık bir parçada birkaç binanın davranışı `other` sonucunu belirliyor. Final modelin `other` PR-AUC'sinin seed'ler arası std'si testte 0.029, ortalamanın yarısından fazla.

## 8. Notlar

- **IF:** yalnız literatür referansı; seçime girmez
- **L1-FE:** G7 içeren modellerin val değerleri hafif iyimser olabilir (L4-semi val etiketiyle erken durduruldu); test etkilenmez
- **L1-OFF:** OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez; G7 içeren modellerin val değerleri hafif iyimser olabilir (L4-semi val etiketiyle erken durduruldu); test etkilenmez
- **L2-FE:** G7 içeren modellerin val değerleri hafif iyimser olabilir (L4-semi val etiketiyle erken durduruldu); test etkilenmez
- **L4-semi:** val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür); kullanıcı kabulü, test etkilenmez
- **L5-FE:** L1-FE + L2-FE; ağırlık val genel PR-AUC ile; G7 içeren modellerin val değerleri hafif iyimser olabilir (L4-semi val etiketiyle erken durduruldu); test etkilenmez
- L3 (CNN+GRU) eğitilmedi (§17). Site bölmesinde test yapılmadı (yalnız val sağlamlık kontrolüydü).
