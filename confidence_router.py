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