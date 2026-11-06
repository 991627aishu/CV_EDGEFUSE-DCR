1RUA24CSE0243, 1RUA24CSE0263, 1RUA24CSE0270, 1RUA24CSE0308
# EdgeFuse-DCR: Comprehensive Comparative Evaluation

| Pipeline Stage / Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Boundary F1 | Grad MAE | Latency (ms) | Peak RAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-Nearest** | 0.1556 | 0.0394 | 36.16 | 0.9583 | 0.7363 | 0.2237 | 16.7 | 13.50 |
| **A0-Bilinear** | 0.1274 | 0.0393 | 37.90 | 0.9740 | 0.8426 | 0.1655 | 14.2 | 13.50 |
| **A0-Bicubic** | 0.1327 | 0.0433 | 37.55 | 0.9717 | 0.8044 | 0.1818 | 12.9 | 13.50 |
| **A1-JointBilateral** | 0.1215 | 0.0505 | 38.31 | 0.9786 | 0.7634 | 0.1507 | 59.8 | 20.24 |
| **A1-GuidedFilter** | 0.1264 | 0.0554 | 37.96 | 0.9777 | 0.7479 | 0.1523 | 54.3 | 21.11 |
| **A2-CNN-Alone** | 0.1272 | 0.0404 | 37.91 | 0.9738 | 0.8421 | 0.1664 | 2486.0 | 23.75 |
| **A3-CNN+Confidence** | 0.1272 | 0.0400 | 37.91 | 0.9739 | 0.8421 | 0.1660 | 2806.2 | 23.46 |
| **A4-CNN+Conf+Boundary** | 0.1143 | 0.0433 | 38.84 | 0.9791 | 0.8162 | 0.1482 | 3279.8 | 37.82 |
| **A5-EdgeFuse-DCR** | 0.1143 | 0.0433 | 38.84 | 0.9791 | 0.8161 | 0.1482 | 2852.3 | 38.99 |

### KITTI Outdoor Benchmark (Validation Subset - LiDAR 4% Sparsity)

| Pipeline Stage / Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Boundary F1 | Grad MAE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-Nearest** | 1.3956 | 0.2726 | 35.17 | 0.1317 | 0.7207 | 31.5591 | 85.7 |
| **A0-Bilinear** | 1.3956 | 0.2726 | 35.17 | 0.1317 | 0.7207 | 31.5591 | 97.2 |
| **A1-GuidedFilter** | 1.0594 | 0.3010 | 37.56 | 0.1316 | 0.5597 | 31.4948 | 317.1 |
| **A2-CNN-Alone** | 1.4331 | 0.3585 | 34.94 | 0.1323 | 0.7316 | 31.4500 | 4337.7 |
| **A5-EdgeFuse-DCR** | 1.1732 | 0.3334 | 36.67 | 0.1324 | 0.7113 | 31.4812 | 3866.3 |

### Middlebury Stereo 2021 Benchmark (Artroom1 - Metric Disparity Inversion)

| Pipeline Stage / Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Boundary F1 | Grad MAE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-Nearest** | 0.5472 | 0.1008 | 33.20 | 0.8210 | 0.6989 | 2.1190 | 86.7 |
| **A0-Bilinear** | 0.4656 | 0.1066 | 34.60 | 0.8253 | 0.6843 | 2.0328 | 90.9 |
| **A1-GuidedFilter** | 0.4634 | 0.1581 | 34.64 | 0.8229 | 0.5292 | 2.0509 | 260.4 |
| **A2-CNN-Alone** | 0.4805 | 0.1881 | 34.32 | 0.8197 | 0.7067 | 2.1528 | 4619.1 |
| **A5-EdgeFuse-DCR** | 0.4468 | 0.1955 | 34.96 | 0.8225 | 0.6653 | 2.0934 | 5420.3 |

### Controlled Texture-Copying Ablation (High-Contrast Wallpaper on Planar Surface)

| Pipeline Stage / Method | RMSE (m) | MAE (m) | PSNR (dB) | Gradient Error | Qualitative Texture Artifact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A0: Bicubic Baseline** | 0.0175 | 0.0139 | 55.15 | 0.08045 | Purely blurred, no boundary definition |
| **A1: Guided Filter** | 0.0070 | 0.0056 | 63.14 | 0.01463 | Severe texture copying (checkerboard relief transferred to depth) |
| **A2: CNN Direct Prediction** | 0.0138 | 0.0109 | 57.17 | 0.05947 | Coarse residual artifact |
| **A3: CNN + Confidence Blending** | 0.0137 | 0.0108 | 57.26 | 0.05891 | Mild smoothing via confidence weights |
| **A4: EdgeFuse w/o Texture Gate** | 0.0117 | 0.0093 | 58.65 | 0.04640 | Boundary preserved but slight texture embossment |
| **A5: Full EdgeFuse-DCR (Hybrid)** | **0.0117** | **0.0092** | **58.67** | **0.04625** | Texture-copying suppressed; genuine boundaries preserved |

