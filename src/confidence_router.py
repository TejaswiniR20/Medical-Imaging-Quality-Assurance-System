# confidence_router.py — Confidence Routing for Pneumonia Detection

def route_prediction(pneumonia_prob: float, high_threshold=0.85, low_threshold=0.15):
    """
    Routes a prediction based on model confidence.

    The model outputs a single sigmoid value (pneumonia probability).
    Normal probability = 1 - pneumonia_prob.

    Args:
        pneumonia_prob: float in [0, 1] — sigmoid output from the model
        high_threshold: above this → Pneumonia (high confidence)
        low_threshold:  below this → Normal (high confidence)
        between the two → flagged for expert review

    Returns:
        dict with predicted class, confidence score, and routing decision
    """
    normal_prob = 1.0 - pneumonia_prob

    if pneumonia_prob >= high_threshold:
        predicted_class = "Pneumonia"
        confidence      = pneumonia_prob
        decision        = "Automated"
    elif pneumonia_prob <= low_threshold:
        predicted_class = "Normal"
        confidence      = normal_prob
        decision        = "Automated"
    else:
        predicted_class = "Pneumonia" if pneumonia_prob >= 0.5 else "Normal"
        confidence      = max(pneumonia_prob, normal_prob)
        decision        = "Review"

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 4),
        "pneumonia_prob":  round(pneumonia_prob, 4),
        "normal_prob":     round(normal_prob, 4),
        "decision":        decision,
        "needs_review":    decision == "Review",
    }
