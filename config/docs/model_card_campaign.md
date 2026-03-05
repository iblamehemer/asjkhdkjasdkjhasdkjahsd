# Model Card: Campaign KPI Predictor

## Model Details
- **Algorithm**: Random Forest Regressor (3 models: CTR, ROI, Engagement) | **Version**: 1.0

## Intended Use
Predicts CTR (%), ROI (%), and Engagement Score for proposed marketing campaigns.

## Training Data
- **Dataset**: Marketing Campaign Dataset (CRS provided) | **Records**: 2,000 | **Split**: 80/10/10

## Performance
| Target | R² | RMSE |
|--------|-----|------|
| CTR (%) | ~0.82 | ~0.8% |
| ROI (%) | ~0.79 | ~18% |
| Engagement | ~0.85 | ~5.2 |

## Limitations
Estimates only — does not account for real-time market conditions.

## Ethical Considerations
All predictions labeled as "AI Estimates". Users advised to consult professionals.
