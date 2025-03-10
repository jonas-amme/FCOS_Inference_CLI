import json

from typer.testing import CliRunner
from ..main import app

from pathlib import Path 


runner = CliRunner()


def test_detect():
    """Test the `detect` command for single image inference."""

    # Setup test case 
    input_path = Path('./test/test_patch.tiff')
    output_dir = Path('./test')
    config_path = Path('./configs/FCOS_18.yaml')

    # Run detect command 
    result = runner.invoke(
        app,
        [
            "detect",
            str(config_path),
            str(input_path),
            str(output_dir),
        ],
    )

    # Assert succesful execution
    assert result.exit_code == 0, f"CLI failed with error: {result.output}"

    # Check that the output directory contains the expected JSON result
    output_file = Path('./test/test_patch_detections.json')
    assert output_file.exists(), "No output JSON file was created!"

    # Verify the contents of the JSON file
    with open(output_file, "r") as f:
        data = json.load(f)
        assert "boxes" in data, "Output JSON is missing 'boxes' key!"
        assert "scores" in data, "Output JSON is missing 'scores' key!"
        assert "labels" in data, "Output JSON is missing 'labels' key!"
        assert "image_path" in data, "Output JSON is missing 'image_path' key!"
        assert data["image_path"] == str(input_path), "Image path in JSON is incorrect!"


    # Verify that the predictions are the same
    test_file = "./test/test_patch_detection_precomputed.json"
    test_data = json.load(open(test_file, "r"))
    pred_data = json.load(open(output_file, "r"))

    for key in ["boxes", "scores", "labels"]:
        test_values = test_data[key]
        pred_values = pred_data[key]
        
        assert len(test_values) == len(pred_values), f"Different number of values {key} key!"
        assert all([test_values[i] == pred_values[i] for i in range(len(test_values))]), f"Values are not the same {key} key!"

