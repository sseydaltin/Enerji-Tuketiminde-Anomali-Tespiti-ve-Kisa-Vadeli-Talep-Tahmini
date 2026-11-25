# LEAD final val özeti (bina bölmesi, 3 seed ort. ± std) — test DEĞİL

Birincil metrik: `other` (davranışsal) dilimi PR-AUC (protokol §18). Taban = dilimin pozitif oranı (rastgele skorun beklenen PR-AUC'si). L3 eğitilmedi (§17). Kaynak: results/lead/tables/val_results.csv.

## Ana tablo (genel ROC-AUC önde; literatür kıyası için)

| model | set | genel ROC-AUC | genel PR-AUC | other PR-AUC | other F1 | other olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 0.938 ± 0.035 | 0.707 ± 0.109 | 0.057 ± 0.032 | 0.069 ± 0.060 | 0.240 ± 0.142 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L5-FE | full | 0.914 ± 0.036 | 0.668 ± 0.101 | 0.053 ± 0.031 | 0.083 ± 0.101 | 0.210 ± 0.193 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | full | 0.912 ± 0.033 | 0.667 ± 0.099 | 0.053 ± 0.030 | 0.080 ± 0.096 | 0.214 ± 0.200 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2-FE | full | 0.903 ± 0.037 | 0.594 ± 0.086 | 0.045 ± 0.019 | 0.076 ± 0.064 | 0.172 ± 0.103 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | full | 0.898 ± 0.028 | 0.627 ± 0.089 | 0.029 ± 0.012 | 0.027 ± 0.038 | 0.156 ± 0.132 |  |
| L5-FE | behavior | 0.872 ± 0.015 | 0.583 ± 0.038 | 0.074 ± 0.057 | 0.114 ± 0.111 | 0.236 ± 0.211 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | behavior | 0.863 ± 0.016 | 0.572 ± 0.035 | 0.055 ± 0.039 | 0.083 ± 0.085 | 0.190 ± 0.188 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | behavior | 0.857 ± 0.005 | 0.575 ± 0.017 | 0.058 ± 0.011 | 0.104 ± 0.025 | 0.178 ± 0.049 |  |
| L2-FE | behavior | 0.852 ± 0.015 | 0.516 ± 0.086 | 0.075 ± 0.062 | 0.120 ± 0.116 | 0.212 ± 0.115 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | full | 0.838 ± 0.026 | 0.520 ± 0.098 | 0.028 ± 0.011 | 0.070 ± 0.033 | 0.170 ± 0.060 |  |
| L2 | behavior | 0.830 ± 0.028 | 0.499 ± 0.085 | 0.032 ± 0.015 | 0.072 ± 0.041 | 0.158 ± 0.048 |  |
| IF | behavior | 0.752 ± 0.023 | 0.089 ± 0.051 | 0.012 ± 0.003 | 0.028 ± 0.012 | 0.145 ± 0.094 | yalnız literatür referansı; seçime girmez |
| L0-R2 | - | 0.749 ± 0.054 | 0.254 ± 0.144 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.006 ± 0.006 |  |
| L0-R3 | - | 0.726 ± 0.051 | 0.225 ± 0.154 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.082 ± 0.027 |  |
| L0-A0 | - | 0.720 ± 0.028 | 0.361 ± 0.011 | 0.057 ± 0.006 | 0.151 ± 0.009 | 0.339 ± 0.010 |  |
| L4-semi | - | 0.718 ± 0.023 | 0.431 ± 0.035 | 0.048 ± 0.015 | 0.108 ± 0.042 | 0.327 ± 0.114 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L4 | - | 0.601 ± 0.013 | 0.140 ± 0.039 | 0.055 ± 0.022 | 0.135 ± 0.046 | 0.539 ± 0.089 |  |
| L0-R1 | - | 0.529 ± 0.003 | 0.054 ± 0.009 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.080 ± 0.028 |  |
| L0-normal | - | 0.500 ± 0.000 | 0.023 ± 0.002 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

Taban (genel PR-AUC) ≈ 0.0230; taban (other PR-AUC) ≈ 0.0073.

## Literatür değerleri (yan yana; koşullar farklı)

| kaynak | genel ROC-AUC | koşul farkları (bizim protokolümüze göre) |
|---|---|---|
| Kaggle LEAD birincisi (Fu, Arjunan & Miller 2022 [2]) | 0,9866 | 206 görülmemiş bina, etiketleri gizli Kaggle testi (biz: 200 etiketli binayı 120/40/40 bölüyoruz, val); nedensel kısıt yok, iki yönlü öznitelik serbest (biz: ana sonuç nedensel); private LB; yalnız ROC-AUC |
| Tech Science yayını (kullanıcının verdiği değer) | 0,952 | künye, bölme ve öznitelik koşulları repoda kayıtlı değil — DOĞRULANMALI |

Kıyas yalnız bu farklar yazılarak yapılır; "daha iyi / daha kötü" doğrudan söylenmez (§14). Bizim satırlarımızdan literatürle en yakın koşuldaki L1-OFF'tur (iki yönlü öznitelik, §19); ana sonuç çevrimiçidir (nedensel).

## other (PR-AUC taban ≈ 0.0073)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L2-FE | behavior | 3 | 0.735 ± 0.081 | 0.075 ± 0.062 | 0.120 ± 0.116 | 0.212 ± 0.115 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | behavior | 3 | 0.763 ± 0.061 | 0.074 ± 0.057 | 0.114 ± 0.111 | 0.236 ± 0.211 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1 | behavior | 3 | 0.745 ± 0.024 | 0.058 ± 0.011 | 0.104 ± 0.025 | 0.178 ± 0.049 |  |
| L1-OFF | full | 3 | 0.823 ± 0.086 | 0.057 ± 0.032 | 0.069 ± 0.060 | 0.240 ± 0.142 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L0-A0 | - | 3 | 0.625 ± 0.045 | 0.057 ± 0.006 | 0.151 ± 0.009 | 0.339 ± 0.010 |  |
| L1-FE | behavior | 3 | 0.742 ± 0.049 | 0.055 ± 0.039 | 0.083 ± 0.085 | 0.190 ± 0.188 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L4 | - | 3 | 0.640 ± 0.046 | 0.055 ± 0.022 | 0.135 ± 0.046 | 0.539 ± 0.089 |  |
| L5-FE | full | 3 | 0.776 ± 0.083 | 0.053 ± 0.031 | 0.083 ± 0.101 | 0.210 ± 0.193 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | full | 3 | 0.770 ± 0.074 | 0.053 ± 0.030 | 0.080 ± 0.096 | 0.214 ± 0.200 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L4-semi | - | 3 | 0.648 ± 0.046 | 0.048 ± 0.015 | 0.108 ± 0.042 | 0.327 ± 0.114 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L2-FE | full | 3 | 0.772 ± 0.090 | 0.045 ± 0.019 | 0.076 ± 0.064 | 0.172 ± 0.103 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | behavior | 3 | 0.634 ± 0.050 | 0.032 ± 0.015 | 0.072 ± 0.041 | 0.158 ± 0.048 |  |
| L1 | full | 3 | 0.739 ± 0.044 | 0.029 ± 0.012 | 0.027 ± 0.038 | 0.156 ± 0.132 |  |
| L2 | full | 3 | 0.639 ± 0.038 | 0.028 ± 0.011 | 0.070 ± 0.033 | 0.170 ± 0.060 |  |
| IF | behavior | 3 | 0.638 ± 0.054 | 0.012 ± 0.003 | 0.028 ± 0.012 | 0.145 ± 0.094 | yalnız literatür referansı; seçime girmez |
| L0-R1 | - | 3 | 0.499 ± 0.000 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.080 ± 0.028 |  |
| L0-R3 | - | 3 | 0.490 ± 0.010 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.082 ± 0.027 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| L0-R2 | - | 3 | 0.454 ± 0.015 | 0.007 ± 0.002 | 0.000 ± 0.000 | 0.006 ± 0.006 |  |

## genel (PR-AUC taban ≈ 0.0230)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 3 | 0.938 ± 0.035 | 0.707 ± 0.109 | 0.680 ± 0.095 | 0.665 ± 0.158 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L5-FE | full | 3 | 0.914 ± 0.036 | 0.668 ± 0.101 | 0.649 ± 0.075 | 0.504 ± 0.133 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | full | 3 | 0.912 ± 0.033 | 0.667 ± 0.099 | 0.649 ± 0.076 | 0.507 ± 0.138 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | full | 3 | 0.898 ± 0.028 | 0.627 ± 0.089 | 0.645 ± 0.061 | 0.525 ± 0.088 |  |
| L2-FE | full | 3 | 0.903 ± 0.037 | 0.594 ± 0.086 | 0.597 ± 0.090 | 0.356 ± 0.109 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | behavior | 3 | 0.872 ± 0.015 | 0.583 ± 0.038 | 0.593 ± 0.048 | 0.328 ± 0.112 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1 | behavior | 3 | 0.857 ± 0.005 | 0.575 ± 0.017 | 0.604 ± 0.025 | 0.321 ± 0.029 |  |
| L1-FE | behavior | 3 | 0.863 ± 0.016 | 0.572 ± 0.035 | 0.597 ± 0.041 | 0.315 ± 0.097 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | full | 3 | 0.838 ± 0.026 | 0.520 ± 0.098 | 0.560 ± 0.080 | 0.411 ± 0.032 |  |
| L2-FE | behavior | 3 | 0.852 ± 0.015 | 0.516 ± 0.086 | 0.551 ± 0.082 | 0.321 ± 0.066 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | behavior | 3 | 0.830 ± 0.028 | 0.499 ± 0.085 | 0.545 ± 0.062 | 0.301 ± 0.049 |  |
| L4-semi | - | 3 | 0.718 ± 0.023 | 0.431 ± 0.035 | 0.502 ± 0.054 | 0.353 ± 0.065 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L0-A0 | - | 3 | 0.720 ± 0.028 | 0.361 ± 0.011 | 0.423 ± 0.020 | 0.279 ± 0.005 |  |
| L0-R2 | - | 3 | 0.749 ± 0.054 | 0.254 ± 0.144 | 0.400 ± 0.172 | 0.223 ± 0.024 |  |
| L0-R3 | - | 3 | 0.726 ± 0.051 | 0.225 ± 0.154 | 0.432 ± 0.178 | 0.513 ± 0.037 |  |
| L4 | - | 3 | 0.601 ± 0.013 | 0.140 ± 0.039 | 0.206 ± 0.051 | 0.460 ± 0.063 |  |
| IF | behavior | 3 | 0.752 ± 0.023 | 0.089 ± 0.051 | 0.204 ± 0.054 | 0.261 ± 0.040 | yalnız literatür referansı; seçime girmez |
| L0-R1 | - | 3 | 0.529 ± 0.003 | 0.054 ± 0.009 | 0.105 ± 0.009 | 0.383 ± 0.031 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.023 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## gap_edge (PR-AUC taban ≈ 0.0026)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 3 | 0.997 ± 0.002 | 0.605 ± 0.250 | 0.523 ± 0.180 | 0.769 ± 0.172 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L1-FE | full | 3 | 0.945 ± 0.012 | 0.505 ± 0.089 | 0.456 ± 0.082 | 0.486 ± 0.150 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | full | 3 | 0.903 ± 0.001 | 0.502 ± 0.049 | 0.476 ± 0.125 | 0.532 ± 0.078 |  |
| L5-FE | full | 3 | 0.944 ± 0.011 | 0.485 ± 0.068 | 0.445 ± 0.090 | 0.481 ± 0.143 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L5-FE | behavior | 3 | 0.858 ± 0.028 | 0.294 ± 0.122 | 0.288 ± 0.110 | 0.143 ± 0.115 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1 | behavior | 3 | 0.807 ± 0.047 | 0.292 ± 0.051 | 0.303 ± 0.037 | 0.138 ± 0.041 |  |
| L0-R1 | - | 3 | 0.762 ± 0.005 | 0.291 ± 0.062 | 0.533 ± 0.060 | 0.522 ± 0.007 |  |
| L1-FE | behavior | 3 | 0.839 ± 0.034 | 0.288 ± 0.110 | 0.306 ± 0.125 | 0.125 ± 0.105 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | full | 3 | 0.902 ± 0.035 | 0.277 ± 0.165 | 0.281 ± 0.114 | 0.333 ± 0.022 |  |
| L4-semi | - | 3 | 0.763 ± 0.021 | 0.265 ± 0.021 | 0.319 ± 0.065 | 0.196 ± 0.065 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L2-FE | full | 3 | 0.895 ± 0.006 | 0.183 ± 0.051 | 0.217 ± 0.078 | 0.224 ± 0.144 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2-FE | behavior | 3 | 0.857 ± 0.020 | 0.181 ± 0.153 | 0.227 ± 0.160 | 0.152 ± 0.081 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | behavior | 3 | 0.839 ± 0.031 | 0.176 ± 0.091 | 0.196 ± 0.060 | 0.134 ± 0.040 |  |
| L4 | - | 3 | 0.760 ± 0.050 | 0.169 ± 0.031 | 0.159 ± 0.064 | 0.293 ± 0.091 |  |
| L0-A0 | - | 3 | 0.578 ± 0.015 | 0.105 ± 0.029 | 0.124 ± 0.036 | 0.078 ± 0.025 |  |
| L0-R3 | - | 3 | 0.790 ± 0.016 | 0.073 ± 0.053 | 0.189 ± 0.126 | 0.525 ± 0.006 |  |
| IF | behavior | 3 | 0.604 ± 0.047 | 0.006 ± 0.004 | 0.020 ± 0.011 | 0.087 ± 0.024 | yalnız literatür referansı; seçime girmez |
| L0-R2 | - | 3 | 0.512 ± 0.014 | 0.006 ± 0.002 | 0.032 ± 0.025 | 0.003 ± 0.001 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.003 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## frozen (PR-AUC taban ≈ 0.0134)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 3 | 0.995 ± 0.002 | 0.903 ± 0.037 | 0.808 ± 0.052 | 0.914 ± 0.043 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L5-FE | full | 3 | 0.991 ± 0.004 | 0.884 ± 0.054 | 0.803 ± 0.048 | 0.921 ± 0.049 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | full | 3 | 0.990 ± 0.005 | 0.882 ± 0.054 | 0.806 ± 0.047 | 0.918 ± 0.043 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | full | 3 | 0.989 ± 0.005 | 0.859 ± 0.067 | 0.804 ± 0.034 | 0.951 ± 0.055 |  |
| L1-FE | behavior | 3 | 0.946 ± 0.069 | 0.803 ± 0.104 | 0.781 ± 0.069 | 0.916 ± 0.058 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | behavior | 3 | 0.946 ± 0.070 | 0.797 ± 0.092 | 0.759 ± 0.032 | 0.867 ± 0.038 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L2-FE | full | 3 | 0.980 ± 0.007 | 0.792 ± 0.018 | 0.731 ± 0.039 | 0.878 ± 0.080 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | behavior | 3 | 0.937 ± 0.051 | 0.791 ± 0.085 | 0.775 ± 0.040 | 0.921 ± 0.018 |  |
| L2 | full | 3 | 0.938 ± 0.013 | 0.697 ± 0.068 | 0.669 ± 0.098 | 0.868 ± 0.034 |  |
| L2 | behavior | 3 | 0.942 ± 0.017 | 0.693 ± 0.063 | 0.685 ± 0.033 | 0.854 ± 0.050 |  |
| L2-FE | behavior | 3 | 0.925 ± 0.070 | 0.683 ± 0.062 | 0.676 ± 0.069 | 0.843 ± 0.049 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L4-semi | - | 3 | 0.758 ± 0.061 | 0.560 ± 0.043 | 0.630 ± 0.018 | 0.751 ± 0.067 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L0-A0 | - | 3 | 0.803 ± 0.049 | 0.474 ± 0.044 | 0.510 ± 0.018 | 0.676 ± 0.038 |  |
| L0-R2 | - | 3 | 0.961 ± 0.017 | 0.404 ± 0.189 | 0.504 ± 0.219 | 0.992 ± 0.004 |  |
| L0-R3 | - | 3 | 0.843 ± 0.028 | 0.290 ± 0.184 | 0.485 ± 0.212 | 0.993 ± 0.003 |  |
| IF | behavior | 3 | 0.850 ± 0.033 | 0.088 ± 0.062 | 0.190 ± 0.064 | 0.808 ± 0.064 | yalnız literatür referansı; seçime girmez |
| L4 | - | 3 | 0.550 ± 0.014 | 0.076 ± 0.031 | 0.138 ± 0.051 | 0.755 ± 0.044 |  |
| L0-R1 | - | 3 | 0.499 ± 0.000 | 0.013 ± 0.003 | 0.000 ± 0.000 | 0.416 ± 0.060 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.013 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## paylasilan (PR-AUC taban ≈ 0.0145)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 3 | 0.968 ± 0.014 | 0.790 ± 0.075 | 0.738 ± 0.081 | 0.756 ± 0.136 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L1-FE | full | 3 | 0.948 ± 0.017 | 0.749 ± 0.073 | 0.718 ± 0.044 | 0.560 ± 0.140 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | full | 3 | 0.949 ± 0.018 | 0.748 ± 0.070 | 0.714 ± 0.038 | 0.557 ± 0.135 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1 | full | 3 | 0.929 ± 0.011 | 0.690 ± 0.064 | 0.692 ± 0.017 | 0.575 ± 0.086 |  |
| L2-FE | full | 3 | 0.936 ± 0.013 | 0.661 ± 0.050 | 0.637 ± 0.059 | 0.384 ± 0.124 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | behavior | 3 | 0.905 ± 0.019 | 0.650 ± 0.027 | 0.659 ± 0.035 | 0.353 ± 0.128 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | behavior | 3 | 0.894 ± 0.024 | 0.636 ± 0.024 | 0.662 ± 0.030 | 0.331 ± 0.112 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | behavior | 3 | 0.884 ± 0.018 | 0.635 ± 0.027 | 0.665 ± 0.013 | 0.345 ± 0.050 |  |
| L2 | full | 3 | 0.896 ± 0.020 | 0.599 ± 0.098 | 0.602 ± 0.103 | 0.457 ± 0.042 |  |
| L2-FE | behavior | 3 | 0.900 ± 0.020 | 0.577 ± 0.108 | 0.597 ± 0.108 | 0.354 ± 0.087 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | behavior | 3 | 0.892 ± 0.018 | 0.574 ± 0.078 | 0.598 ± 0.055 | 0.329 ± 0.047 |  |
| L4-semi | - | 3 | 0.745 ± 0.039 | 0.481 ± 0.048 | 0.555 ± 0.054 | 0.367 ± 0.073 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L0-A0 | - | 3 | 0.739 ± 0.024 | 0.372 ± 0.016 | 0.429 ± 0.029 | 0.256 ± 0.021 |  |
| L0-R2 | - | 3 | 0.791 ± 0.030 | 0.231 ± 0.119 | 0.389 ± 0.176 | 0.210 ± 0.015 |  |
| L0-R3 | - | 3 | 0.769 ± 0.029 | 0.217 ± 0.141 | 0.420 ± 0.185 | 0.552 ± 0.029 |  |
| L4 | - | 3 | 0.597 ± 0.015 | 0.122 ± 0.036 | 0.192 ± 0.053 | 0.459 ± 0.075 |  |
| IF | behavior | 3 | 0.754 ± 0.026 | 0.066 ± 0.051 | 0.157 ± 0.061 | 0.249 ± 0.026 | yalnız literatür referansı; seçime girmez |
| L0-R1 | - | 3 | 0.538 ± 0.002 | 0.051 ± 0.012 | 0.131 ± 0.007 | 0.458 ± 0.023 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.015 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## tekil (PR-AUC taban ≈ 0.0088)

| model | set | n seed | ROC-AUC | PR-AUC | F1 | olay recall | açıklama |
|---|---|---|---|---|---|---|---|
| L1-OFF | full | 3 | 0.892 ± 0.063 | 0.416 ± 0.137 | 0.429 ± 0.124 | 0.477 ± 0.156 | OFFLINE (geleceğe bakan öznitelikler); yalnız literatür kıyası, seçime girmez |
| L5-FE | full | 3 | 0.862 ± 0.060 | 0.390 ± 0.099 | 0.408 ± 0.058 | 0.392 ± 0.106 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L1-FE | full | 3 | 0.857 ± 0.053 | 0.387 ± 0.094 | 0.407 ± 0.055 | 0.394 ± 0.110 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L1 | full | 3 | 0.852 ± 0.050 | 0.380 ± 0.067 | 0.439 ± 0.034 | 0.415 ± 0.064 |  |
| L1 | behavior | 3 | 0.813 ± 0.022 | 0.361 ± 0.055 | 0.400 ± 0.052 | 0.274 ± 0.069 |  |
| L1-FE | behavior | 3 | 0.810 ± 0.021 | 0.350 ± 0.037 | 0.402 ± 0.041 | 0.282 ± 0.069 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L5-FE | behavior | 3 | 0.816 ± 0.031 | 0.341 ± 0.040 | 0.384 ± 0.034 | 0.278 ± 0.090 | L1-FE + L2-FE; ağırlık val genel PR-AUC ile |
| L2-FE | full | 3 | 0.852 ± 0.072 | 0.260 ± 0.085 | 0.346 ± 0.102 | 0.296 ± 0.079 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L4-semi | - | 3 | 0.674 ± 0.024 | 0.244 ± 0.054 | 0.328 ± 0.061 | 0.331 ± 0.064 | val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketini görür) |
| L2-FE | behavior | 3 | 0.774 ± 0.029 | 0.240 ± 0.094 | 0.312 ± 0.091 | 0.253 ± 0.049 | G7 seçilen seed'lerde L4-semi OOF artığı (val değeri val-ES'li L4-semi'den) |
| L2 | full | 3 | 0.745 ± 0.033 | 0.229 ± 0.120 | 0.312 ± 0.091 | 0.311 ± 0.026 |  |
| L2 | behavior | 3 | 0.732 ± 0.042 | 0.225 ± 0.123 | 0.300 ± 0.076 | 0.240 ± 0.062 |  |
| L0-A0 | - | 3 | 0.687 ± 0.043 | 0.218 ± 0.032 | 0.293 ± 0.049 | 0.327 ± 0.038 |  |
| L0-R2 | - | 3 | 0.686 ± 0.084 | 0.124 ± 0.104 | 0.228 ± 0.173 | 0.254 ± 0.047 |  |
| L0-R3 | - | 3 | 0.660 ± 0.078 | 0.084 ± 0.089 | 0.233 ± 0.168 | 0.432 ± 0.035 |  |
| L4 | - | 3 | 0.605 ± 0.015 | 0.062 ± 0.021 | 0.135 ± 0.043 | 0.471 ± 0.059 |  |
| IF | behavior | 3 | 0.748 ± 0.045 | 0.033 ± 0.012 | 0.083 ± 0.018 | 0.290 ± 0.107 | yalnız literatür referansı; seçime girmez |
| L0-R1 | - | 3 | 0.514 ± 0.002 | 0.014 ± 0.000 | 0.049 ± 0.006 | 0.220 ± 0.007 |  |
| L0-normal | - | 3 | 0.500 ± 0.000 | 0.009 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## Seçim

- §18.2 kuralı (geçerli): **L5-FE / full**
- §12 kuralı (eski, tarihçe için): L1-FE / full
