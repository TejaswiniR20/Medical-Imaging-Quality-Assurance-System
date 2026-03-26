import os
import cv2
import numpy as np
import pytest
import sys
import shutil


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.image_enhancement import enhance_image, TARGET_SIZE


TEST_DIR = os.path.join(os.path.dirname(__file__), "temp_test_data")
MOCK_IMAGE_PATH = os.path.join(TEST_DIR, "test_input.jpg")
MOCK_OUTPUT_PATH = os.path.join(TEST_DIR, "test_output.jpg")

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Setup and teardown for mock test image."""
    os.makedirs(TEST_DIR, exist_ok=True)
    
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite(MOCK_IMAGE_PATH, dummy_img)
    
    yield
    

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_enhance_image_exists():
    """Verify enhance_image function works as expected."""
    result = enhance_image(MOCK_IMAGE_PATH, MOCK_OUTPUT_PATH)
    assert result is True
    assert os.path.exists(MOCK_OUTPUT_PATH)

def test_image_dimensions():
    """Verify the output image is resized correctly (224x224)."""
    enhance_image(MOCK_IMAGE_PATH, MOCK_OUTPUT_PATH)
    processed_img = cv2.imread(MOCK_OUTPUT_PATH)
    assert processed_img.shape[0] == TARGET_SIZE[0]
    assert processed_img.shape[1] == TARGET_SIZE[1]

def test_image_is_grayscale():
    """Verify the output is single-channel grayscale."""
    enhance_image(MOCK_IMAGE_PATH, MOCK_OUTPUT_PATH)
    
    processed_img = cv2.imread(MOCK_OUTPUT_PATH, cv2.IMREAD_UNCHANGED)
    
    assert len(processed_img.shape) == 2 or (len(processed_img.shape) == 3 and processed_img.shape[2] == 1)

if __name__ == "__main__":
    
    os.makedirs(TEST_DIR, exist_ok=True)
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite(MOCK_IMAGE_PATH, dummy_img)
    
    print("Running manual verification...")
    if enhance_image(MOCK_IMAGE_PATH, MOCK_OUTPUT_PATH):
        img = cv2.imread(MOCK_OUTPUT_PATH, cv2.IMREAD_UNCHANGED)
        print(f"Result Image Shape: {img.shape}")
        if img.shape[0:2] == (224, 224) and (len(img.shape) == 2):
            print("✅ Size and Grayscale status: [PASS]")
        else:
            print(f"❌ Verification: [FAIL] - Expected (224, 224) and 1 channel, got {img.shape}")
    else:
        print("❌ FAILED to process image.")
    
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
