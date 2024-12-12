import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

# Load data files
def load_data():
    # Input files
    errors_csv = "data_10pct_errors.csv"
    ground_truth_csv = "ground_truth.csv"
    detected_errors_json = "fullMD_10pct.json"
    corrected_csv = "fullMD_10pct.csv"
    corrected_json = "fullMD_10pct_corrections.json"

    # Load CSV files
    errors_df = pd.read_csv(errors_csv).head(2000)
    ground_truth_df = pd.read_csv(ground_truth_csv).head(2000)
    corrected_df = pd.read_csv(corrected_csv).head(2000)

    # Load JSON files
    with open(detected_errors_json, "r") as f:
        detected_errors = json.load(f)
    with open(corrected_json, "r") as f:
        corrections = json.load(f)

    return errors_df, ground_truth_df, detected_errors, corrected_df, corrections, detected_errors_json

# Calculate detection performance
def evaluate_detection(detected_errors, ground_truth_df):
    # Extract actual errors and detected errors
    actual_errors = ground_truth_df.apply(lambda row: row.isnull().any(), axis=1).astype(int)
    detected = pd.Series(
        [1 if "error" in str(item["error_detection"]).lower() else 0 for item in detected_errors["detailed_results"]]
    )

    precision = precision_score(actual_errors, detected)
    recall = recall_score(actual_errors, detected)
    f1 = f1_score(actual_errors, detected)
    accuracy = accuracy_score(actual_errors, detected)

    return precision, recall, f1, accuracy

# Calculate correction performance
def evaluate_correction(corrected_df, ground_truth_df):
    # Compare corrected values to ground truth
    total_correct = (corrected_df.fillna("missing") == ground_truth_df.fillna("missing")).sum().sum()
    total_values = ground_truth_df.size

    accuracy = total_correct / total_values

    return accuracy

# Save results to a CSV file
def save_results_to_csv(results, detected_errors_json):
    # Determine output file name based on detected_errors_json file name
    base_name = os.path.splitext(os.path.basename(detected_errors_json))[0]
    output_csv = f"{base_name}_results.csv"

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

# Main evaluation function
def evaluate_model():
    errors_df, ground_truth_df, detected_errors, corrected_df, corrections, detected_errors_json = load_data()

    print("Evaluating error detection performance...")
    precision, recall, f1, accuracy = evaluate_detection(detected_errors, ground_truth_df)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    print("\nEvaluating error correction performance...")
    correction_accuracy = evaluate_correction(corrected_df, ground_truth_df)
    print(f"Correction Accuracy: {correction_accuracy:.2f}")

    # Save results
    results = {
        "Detection Precision": precision,
        "Detection Recall": recall,
        "Detection F1 Score": f1,
        "Detection Accuracy": accuracy,
        "Correction Accuracy": correction_accuracy
    }
    save_results_to_csv(results, detected_errors_json)

if __name__ == "__main__":
    evaluate_model()
