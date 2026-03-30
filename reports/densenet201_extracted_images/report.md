# DenseNet201 Prediction Report

- Checkpoint: `artifacts/densenet201_pretrained/best_model.pt`
- Input directory: `ingestion/extracted_images`
- Total images: 30

## Overall Distribution

| Predicted class | Count | Mean confidence |
| --- | ---: | ---: |
| Bradycardia_type_II | 10 | 0.6821 |
| DES | 5 | 0.6002 |
| EGJ | 3 | 0.7336 |
| IEM | 11 | 0.9554 |
| Jackhammer | 0 | - |
| normal | 1 | 0.5252 |

## Folder Summary

| Folder | Images | Dominant prediction | Ratio | Mean confidence | Expected class | Match count |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| acalasia_tipo_i | 10 | IEM | 100.00% | 0.9968 | - | - |
| acalasia_tipo_ii | 10 | Bradycardia_type_II | 60.00% | 0.7430 | - | - |
| espasmo_esofagico | 10 | DES | 50.00% | 0.5660 | - | - |
