# YOLOv8 WiderPerson Evaluation Results

## Detection Performance

| Model                     | mAP@50   | mAP@50-95   | Precision   | Recall   |      FPS |   Inference (ms) |   Total Detections |   Avg Det/Image |
|:--------------------------|:---------|:------------|:------------|:---------|---------:|-----------------:|-------------------:|----------------:|
| yolo11x                   | 0.6702   | 0.2700      | 0.7837      | 0.5675   |  18.6937 |          53.4941 |             769661 |        175.6415 |
| Yolov8x                   | 0.6695   | 0.2719      | 0.7784      | 0.5756   |  15.8588 |          63.0565 |             855057 |        195.1294 |
| yolo12l                   | 0.6689   | 0.2694      | 0.7932      | 0.5601   |  23.5834 |          42.4028 |             804899 |        183.6830 |
| yolov9c                   | 0.6658   | 0.2665      | 0.7713      | 0.5785   |  29.2591 |          34.1774 |             963580 |        219.8950 |
| yolo11l                   | 0.6651   | 0.2678      | 0.7834      | 0.5633   |  34.7114 |          28.8089 |             879938 |        200.8074 |
| yolo12m                   | 0.6648   | 0.2686      | 0.7931      | 0.5585   |  32.7519 |          30.5326 |             836798 |        190.9626 |
| yolo12m                   | 0.6648   | 0.2686      | 0.7931      | 0.5585   |  32.6279 |          30.6486 |             836798 |        190.9626 |
| Yolov8l                   | 0.6630   | 0.2668      | 0.7746      | 0.5713   |  23.7063 |          42.1829 |             898226 |        204.9808 |
| yolov9m                   | 0.6606   | 0.2655      | 0.7709      | 0.5709   |  33.1608 |          30.1561 |            1030285 |        235.1175 |
| Yolov8m                   | 0.6562   | 0.2629      | 0.7680      | 0.5736   |  36.0703 |          27.7237 |            1068171 |        243.7634 |
| yolo11m                   | 0.6560   | 0.2636      | 0.7802      | 0.5563   |  42.4716 |          23.5452 |             940592 |        214.6490 |
| yolov10x                  | 0.6485   | 0.2626      | 0.7544      | 0.5557   |  20.2577 |          49.3639 |             902094 |        205.8635 |
| yolov10l                  | 0.6458   | 0.2631      | 0.7612      | 0.5510   |  28.0429 |          35.6597 |             917107 |        209.2896 |
| yolo12s                   | 0.6416   | 0.2571      | 0.7919      | 0.5324   |  51.6823 |          19.3490 |             991417 |        226.2476 |
| yolov9s                   | 0.6392   | 0.2561      | 0.7550      | 0.5563   |  52.9595 |          18.8823 |            1167558 |        266.4441 |
| yolov10m                  | 0.6341   | 0.2584      | 0.7588      | 0.5353   |  42.8355 |          23.3451 |            1008680 |        230.1871 |
| yolo11s                   | 0.6320   | 0.2519      | 0.7625      | 0.5478   |  77.4619 |          12.9096 |            1168177 |        266.5853 |
| Yolov8s                   | 0.6299   | 0.2501      | 0.7457      | 0.5608   |  66.2473 |          15.0950 |            1202335 |        274.3804 |
| yolov9t                   | 0.6133   | 0.2460      | 0.7439      | 0.5314   |  58.0751 |          17.2191 |            1266793 |        289.0901 |
| yolo12n                   | 0.6089   | 0.2458      | 0.7642      | 0.5138   |  58.7463 |          17.0223 |            1215582 |        277.4035 |
| Yolov8n                   | 0.6065   | 0.2424      | 0.7388      | 0.5266   |  95.8470 |          10.4333 |            1282062 |        292.5746 |
| yolo11n                   | 0.6029   | 0.2433      | 0.7489      | 0.5172   |  83.8786 |          11.9220 |            1260269 |        287.6013 |
| yolov10s                  | 0.6024   | 0.2464      | 0.7313      | 0.5064   |  76.6287 |          13.0499 |            1162660 |        265.3263 |
| yolov10n                  | 0.5678   | 0.2381      | 0.7180      | 0.4774   |  94.8368 |          10.5444 |            1260161 |        287.5767 |
| yolo11l_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  98.1335 |          10.1902 |             786908 |        179.5774 |
| yolo11m_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 115.1239 |           8.6863 |             824806 |        188.2259 |
| yolo11n_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 159.7629 |           6.2593 |            1185136 |        270.4555 |
| yolo11s_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 145.8976 |           6.8541 |            1089688 |        248.6737 |
| yolo11x_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  72.7078 |          13.7537 |             682590 |        155.7713 |
| yolo12l_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  68.1483 |          14.6739 |             697477 |        159.1686 |
| yolo12m_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  92.4324 |          10.8187 |             721728 |        164.7029 |
| yolo12n_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 146.9432 |           6.8054 |            1135901 |        259.2198 |
| yolo12s_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 125.8942 |           7.9432 |             899954 |        205.3752 |
| yolov10l_tensorrt_results | N/A      | N/A         | N/A         | N/A      |  61.5856 |          16.2376 |             968902 |        221.1095 |
| yolov10m_tensorrt_results | N/A      | N/A         | N/A         | N/A      | 120.3416 |           8.3097 |             985461 |        224.8884 |
| yolov10n_tensorrt_results | N/A      | N/A         | N/A         | N/A      | 167.2890 |           5.9777 |            1245737 |        284.2850 |
| yolov10s_tensorrt_results | N/A      | N/A         | N/A         | N/A      | 151.0260 |           6.6214 |            1166395 |        266.1787 |
| yolov10x_tensorrt_results | N/A      | N/A         | N/A         | N/A      |  73.0247 |          13.6940 |             864813 |        197.3558 |
| yolov8l_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  82.8038 |          12.0767 |             770606 |        175.8571 |
| yolov8m_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  75.3411 |          13.2730 |            1072686 |        244.7937 |
| yolov8n_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 150.9622 |           6.6242 |            1218363 |        278.0381 |
| yolov8s_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 142.9353 |           6.9962 |            1105108 |        252.1926 |
| yolov8x_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  65.6608 |          15.2298 |             732297 |        167.1148 |
| yolov9c_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  94.0865 |          10.6285 |             839137 |        191.4963 |
| yolov9m_tensorrt_results  | N/A      | N/A         | N/A         | N/A      |  99.4402 |          10.0563 |             904395 |        206.3886 |
| yolov9s_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 131.9293 |           7.5798 |            1076797 |        245.7319 |
| yolov9t_tensorrt_results  | N/A      | N/A         | N/A         | N/A      | 120.8271 |           8.2763 |            1270378 |        289.9083 |

## Inference Speed Details

| Model                     |    FPS |   Mean (ms) |   Std (ms) | Min (ms)   | Max (ms)   | 95th %ile (ms)   |
|:--------------------------|-------:|------------:|-----------:|:-----------|:-----------|:-----------------|
| yolov10n_tensorrt_results | 167.29 |        5.98 |       2.24 | N/A        | N/A        | N/A              |
| yolo11n_tensorrt_results  | 159.76 |        6.26 |       2.13 | N/A        | N/A        | N/A              |
| yolov10s_tensorrt_results | 151.03 |        6.62 |       2.14 | N/A        | N/A        | N/A              |
| yolov8n_tensorrt_results  | 150.96 |        6.62 |       2.10 | N/A        | N/A        | N/A              |
| yolo12n_tensorrt_results  | 146.94 |        6.81 |       2.09 | N/A        | N/A        | N/A              |
| yolo11s_tensorrt_results  | 145.90 |        6.85 |       2.03 | N/A        | N/A        | N/A              |
| yolov8s_tensorrt_results  | 142.94 |        7.00 |       2.09 | N/A        | N/A        | N/A              |
| yolov9s_tensorrt_results  | 131.93 |        7.58 |       2.02 | N/A        | N/A        | N/A              |
| yolo12s_tensorrt_results  | 125.89 |        7.94 |       2.12 | N/A        | N/A        | N/A              |
| yolov9t_tensorrt_results  | 120.83 |        8.28 |       2.16 | N/A        | N/A        | N/A              |
| yolov10m_tensorrt_results | 120.34 |        8.31 |       2.14 | N/A        | N/A        | N/A              |
| yolo11m_tensorrt_results  | 115.12 |        8.69 |       2.17 | N/A        | N/A        | N/A              |
| yolov9m_tensorrt_results  |  99.44 |       10.06 |       2.19 | N/A        | N/A        | N/A              |
| yolo11l_tensorrt_results  |  98.13 |       10.19 |       2.13 | N/A        | N/A        | N/A              |
| Yolov8n                   |  95.85 |       10.43 |       5.96 | 6.84       | 70.78      | 12.65            |
| yolov10n                  |  94.84 |       10.54 |       3.84 | 8.13       | 69.85      | 13.42            |
| yolov9c_tensorrt_results  |  94.09 |       10.63 |       2.14 | N/A        | N/A        | N/A              |
| yolo12m_tensorrt_results  |  92.43 |       10.82 |       2.09 | N/A        | N/A        | N/A              |
| yolo11n                   |  83.88 |       11.92 |       5.47 | 8.44       | 75.45      | 14.31            |
| yolov8l_tensorrt_results  |  82.80 |       12.08 |       2.11 | N/A        | N/A        | N/A              |
| yolo11s                   |  77.46 |       12.91 |       5.33 | 8.48       | 72.19      | 16.01            |
| yolov10s                  |  76.63 |       13.05 |       3.95 | 8.58       | 74.07      | 16.02            |
| yolov8m_tensorrt_results  |  75.34 |       13.27 |       2.17 | N/A        | N/A        | N/A              |
| yolov10x_tensorrt_results |  73.02 |       13.69 |       2.15 | N/A        | N/A        | N/A              |
| yolo11x_tensorrt_results  |  72.71 |       13.75 |       2.22 | N/A        | N/A        | N/A              |
| yolo12l_tensorrt_results  |  68.15 |       14.67 |       2.06 | N/A        | N/A        | N/A              |
| Yolov8s                   |  66.25 |       15.09 |       5.19 | 9.32       | 76.52      | 18.94            |
| yolov8x_tensorrt_results  |  65.66 |       15.23 |       2.10 | N/A        | N/A        | N/A              |
| yolov10l_tensorrt_results |  61.59 |       16.24 |       2.27 | N/A        | N/A        | N/A              |
| yolo12n                   |  58.75 |       17.02 |       6.75 | 13.38      | 84.00      | 19.68            |
| yolov9t                   |  58.08 |       17.22 |       5.08 | 14.13      | 76.55      | 20.53            |
| yolov9s                   |  52.96 |       18.88 |       4.24 | 15.09      | 80.07      | 21.84            |
| yolo12s                   |  51.68 |       19.35 |       7.91 | 13.47      | 104.39     | 31.54            |
| yolov10m                  |  42.84 |       23.35 |       4.41 | 14.24      | 83.27      | 28.02            |
| yolo11m                   |  42.47 |       23.55 |       4.53 | 14.79      | 81.25      | 29.13            |
| Yolov8m                   |  36.07 |       27.72 |       4.92 | 18.08      | 88.09      | 32.97            |
| yolo11l                   |  34.71 |       28.81 |       4.77 | 18.02      | 88.46      | 33.88            |
| yolov9m                   |  33.16 |       30.16 |       5.49 | 18.30      | 90.52      | 36.09            |
| yolo12m                   |  32.75 |       30.53 |       5.41 | 20.32      | 93.84      | 36.35            |
| yolo12m                   |  32.63 |       30.65 |       5.79 | 19.53      | 95.34      | 37.50            |
| yolov9c                   |  29.26 |       34.18 |       5.25 | 21.86      | 97.86      | 39.97            |
| yolov10l                  |  28.04 |       35.66 |       5.20 | 21.60      | 95.37      | 40.82            |
| Yolov8l                   |  23.71 |       42.18 |       5.52 | 26.33      | 101.27     | 48.48            |
| yolo12l                   |  23.58 |       42.40 |       6.43 | 25.79      | 101.74     | 51.16            |
| yolov10x                  |  20.26 |       49.36 |       6.05 | 31.71      | 109.01     | 56.22            |
| yolo11x                   |  18.69 |       53.49 |       7.72 | 33.96      | 173.95     | 62.80            |
| Yolov8x                   |  15.86 |       63.06 |       6.96 | 42.37      | 124.47     | 70.48            |

## Model Details

| Model                     | Type     | Device   |   Batch Size | Model Path      |
|:--------------------------|:---------|:---------|-------------:|:----------------|
| Yolov8l                   | PyTorch  | N/A      |            1 | nan             |
| Yolov8m                   | PyTorch  | N/A      |            1 | nan             |
| Yolov8n                   | PyTorch  | N/A      |            1 | nan             |
| Yolov8s                   | PyTorch  | N/A      |            1 | nan             |
| Yolov8x                   | PyTorch  | N/A      |            1 | nan             |
| yolo11l_tensorrt_results  | TensorRT | 0        |           16 | yolo11l.engine  |
| yolo11l                   | PyTorch  | N/A      |            1 | nan             |
| yolo11m_tensorrt_results  | TensorRT | 0        |           16 | yolo11m.engine  |
| yolo11m                   | PyTorch  | N/A      |            1 | nan             |
| yolo11n_tensorrt_results  | TensorRT | 0        |           16 | yolo11n.engine  |
| yolo11n                   | PyTorch  | N/A      |            1 | nan             |
| yolo11s_tensorrt_results  | TensorRT | 0        |           16 | yolo11s.engine  |
| yolo11s                   | PyTorch  | N/A      |            1 | nan             |
| yolo11x_tensorrt_results  | TensorRT | 0        |           16 | yolo11x.engine  |
| yolo11x                   | PyTorch  | N/A      |            1 | nan             |
| yolo12l_tensorrt_results  | TensorRT | 0        |           16 | yolo12l.engine  |
| yolo12l                   | PyTorch  | N/A      |            1 | nan             |
| yolo12m_tensorrt_results  | TensorRT | 0        |           16 | yolo12m.engine  |
| yolo12m                   | PyTorch  | N/A      |            1 | nan             |
| yolo12m                   | PyTorch  | N/A      |            1 | nan             |
| yolo12n_tensorrt_results  | TensorRT | 0        |           16 | yolo12n.engine  |
| yolo12n                   | PyTorch  | N/A      |            1 | nan             |
| yolo12s_tensorrt_results  | TensorRT | 0        |           16 | yolo12s.engine  |
| yolo12s                   | PyTorch  | N/A      |            1 | nan             |
| yolov10l_tensorrt_results | TensorRT | 0        |           16 | yolov10l.engine |
| yolov10l                  | PyTorch  | N/A      |            1 | nan             |
| yolov10m_tensorrt_results | TensorRT | 0        |           16 | yolov10m.engine |
| yolov10m                  | PyTorch  | N/A      |            1 | nan             |
| yolov10n_tensorrt_results | TensorRT | 0        |           16 | yolov10n.engine |
| yolov10n                  | PyTorch  | N/A      |            1 | nan             |
| yolov10s_tensorrt_results | TensorRT | 0        |           16 | yolov10s.engine |
| yolov10s                  | PyTorch  | N/A      |            1 | nan             |
| yolov10x_tensorrt_results | TensorRT | 0        |           16 | yolov10x.engine |
| yolov10x                  | PyTorch  | N/A      |            1 | nan             |
| yolov8l_tensorrt_results  | TensorRT | 0        |           16 | yolov8l.engine  |
| yolov8m_tensorrt_results  | TensorRT | 0        |           16 | yolov8m.engine  |
| yolov8n_tensorrt_results  | TensorRT | 0        |           16 | yolov8n.engine  |
| yolov8s_tensorrt_results  | TensorRT | 0        |           16 | yolov8s.engine  |
| yolov8x_tensorrt_results  | TensorRT | 0        |           16 | yolov8x.engine  |
| yolov9c_tensorrt_results  | TensorRT | 0        |           16 | yolov9c.engine  |
| yolov9c                   | PyTorch  | N/A      |            1 | nan             |
| yolov9m_tensorrt_results  | TensorRT | 0        |           16 | yolov9m.engine  |
| yolov9m                   | PyTorch  | N/A      |            1 | nan             |
| yolov9s_tensorrt_results  | TensorRT | 0        |           16 | yolov9s.engine  |
| yolov9s                   | PyTorch  | N/A      |            1 | nan             |
| yolov9t_tensorrt_results  | TensorRT | 0        |           16 | yolov9t.engine  |
| yolov9t                   | PyTorch  | N/A      |            1 | nan             |