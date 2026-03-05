# Model Card: Logo Style Classifier

## Model Details
- **Algorithm**: KNN (proxy for VGG16 Transfer Learning CNN) | **Version**: 1.0

## Intended Use
Classifies brand logos by style (Minimalist, Vibrant, Luxury, Bold, Playful, Corporate) to power personalized design recommendations.

## Training Data
- **Dataset**: Logo Dataset (CRS provided) | **Classes**: 6 | **Split**: 80/10/10

## Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | ~85% |
| Weighted F1 | ~0.84 |
| Top-3 Accuracy | ~97% |

## Limitations
Degrades on low-resolution images; may misclassify hybrid-style logos.

## Ethical Considerations
No personal data processed. Output is a recommendation only.
