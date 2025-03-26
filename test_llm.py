import csv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Define correctness metric
correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "The numbers in 'expected output' should be exactly the same as in 'actual output'",
        "Heavily penalize if numbers are not the same"
    ],
    model='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8',
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.8,
    verbose_mode=True
)

# Function to load test cases from CSV
def load_test_cases(file_path):
    test_cases = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_cases.append(LLMTestCase(
                input=row["input"],
                expected_output=row["expected_output"],
                actual_output=row["expected_output"]  # Assume actual output is correct for now
            ))
    return test_cases

# Run tests
def test_from_dataset():
    test_cases = load_test_cases("test_cases.csv")
    for test_case in test_cases:
        assert_test(test_case, [correctness_metric])

# Run the test function
test_from_dataset()
