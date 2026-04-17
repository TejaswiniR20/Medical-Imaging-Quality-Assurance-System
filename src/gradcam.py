# gradcam.py — Grad-CAM Heatmap Generation for MobileNetV2

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image as PILImage

# Last conv layer in MobileNetV2 before global average pooling
GRADCAM_LAYER = "out_relu"


class GradCAM:
    """
    Grad-CAM implementation for the MobileNetV2 Keras model.
    Generates heatmaps highlighting regions that influenced the prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = GRADCAM_LAYER):
        self.model = model
        self.layer_name = layer_name

        # Try to find the correct layer; fall back to last conv layer
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D,
                                      tf.keras.layers.DepthwiseConv2D)):
                    self.layer_name = layer.name
                    break

        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(self.layer_name).output, model.output],
        )

    def generate(self, img_array: np.ndarray) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a preprocessed image.

        Args:
            img_array: preprocessed image, shape (1, 224, 224, 3)

        Returns:
            heatmap: np.ndarray (H, W), values in [0, 1]
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap).numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def overlay_on_image(self, original_image_path: str,
                          heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on the original X-ray image.

        Args:
            original_image_path: path to the original image file
            heatmap: output of generate(), shape (H, W), values in [0, 1]
            alpha: heatmap opacity

        Returns:
            overlaid image as np.ndarray (224, 224, 3), BGR
        """
        img = np.array(PILImage.open(original_image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid

    def overlay_on_array(self, img_bgr: np.ndarray,
                          heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on a BGR image array."""
        img = cv2.resize(img_bgr, (224, 224))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    def save(self, overlaid_image: np.ndarray, output_path: str):
        """Save the overlaid heatmap image to disk."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlaid_image)
