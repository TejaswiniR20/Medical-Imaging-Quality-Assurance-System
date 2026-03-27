# Confidence-Based Prediction Router

## Overview
This module implements a confidence-based routing mechanism for machine learning predictions. It serves as a post-processing layer that interprets model output probabilities and determines whether the prediction is reliable or requires further review.

The primary objectives of this module are:
- Convert raw model outputs into human-readable predictions
- Evaluate prediction confidence
- Flag uncertain predictions for expert validation

---

## Function: `route_prediction`

```python
def route_prediction(probabilities, threshold=0.85):
    """
    probabilities: output from model's softmax layer
                   e.g. [0.92, 0.08] -> [Normal, Pneumonia]
    """
    confidence = max(probabilities)
    predicted_class = probabilities.index(confidence)
    label = "NORMAL" if predicted_class == 0 else "PNEUMONIA"

    if confidence < threshold:
        return {
            "label": label,
            "confidence": confidence,
            "flag": True,
            "reason": "Low confidence — expert review required"
        }
    return {
        "label": label,
        "confidence": confidence,
        "flag": False
    }
```

---

## Parameters

| Parameter     | Type  | Description |
|--------------|-------|------------|
| probabilities | list | Model output probabilities (e.g., `[0.92, 0.08]`) |
| threshold     | float | Confidence threshold (default: 0.85) |

---

## Methodology

1. Identify the maximum probability to determine confidence.
2. Determine the predicted class based on the index:
   - `0 → NORMAL`
   - `1 → PNEUMONIA`
3. Compare confidence with threshold:
   - If confidence < threshold → flag for review
   - Otherwise → accept prediction

---

## Output Format

### High Confidence Prediction
```json
{
  "label": "NORMAL",
  "confidence": 0.92,
  "flag": false
}
```

### Low Confidence Prediction
```json
{
  "label": "PNEUMONIA",
  "confidence": 0.60,
  "flag": true,
  "reason": "Low confidence — expert review required"
}
```

---

## Example Usage

```python
probabilities = [0.78, 0.22]

result = route_prediction(probabilities)

print(result)
```

---

## Use Cases

- Medical image classification (e.g., pneumonia detection)
- Fraud detection systems
- Risk-sensitive decision-making systems
- Machine learning pipelines requiring human oversight

---

## Future Enhancements

- Multi-class classification support
- Dynamic threshold tuning
- Logging and monitoring
- API integration (Flask / FastAPI)
- Confidence calibration methods

---

## File Structure

```
confidence_router.py
README.md
```

---

## Summary

This module adds a reliability layer to machine learning systems by evaluating prediction confidence and routing uncertain outputs for expert review. It is particularly useful in high-stakes domains where prediction accuracy and trust are critical.
