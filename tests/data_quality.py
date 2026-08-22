"""
This checks if the load and clean was well done
"""

import sys
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator

def validate_data(path: str):
    df = pd.read_csv(path)

    # Create an Ephemeral DataContext
    context = gx.get_context(mode="ephemeral")

    # Wrap DataFrame in Validator via a Batch
    batch = Batch(data=df)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[batch],
        data_context=context,
    )

    # Define expectations
    columns = df.columns
    validator.expect_column_values_to_not_be_null(columns)
    validator.expect_column_values_to_be_between("blueWins", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("blueTowersDestroyed", min_value=0, max_value=4)
    validator.expect_column_values_to_be_between("redTowersDestroyed", min_value=0, max_value=4)
    validator.expect_column_values_to_be_between("blueFirstBlood", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("redFirstBlood", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("blueEliteMonsters", min_value=0, max_value=2)
    validator.expect_column_values_to_be_between("blueDragons", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("blueHeralds", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("redEliteMonsters", min_value=0, max_value=2)
    validator.expect_column_values_to_be_between("redDragons", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("redHeralds", min_value=0, max_value=1)

    # Run validation
    results = validator.validate()
    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])
    failed = total - passed

    print(f"\n{path}: {passed}/{total} checks passed")
    if failed:
        print("❌ Failed expectations:")
        for r in results["results"]:
            if not r["success"]:
                config = r["expectation_config"]
                column = config.kwargs.get("column", "N/A")
                expectation_type = config.type
                kwargs = {k: v for k, v in config.kwargs.items() if k != "column"}
                print(f"  - {expectation_type} on column '{column}' with params: {kwargs}")

                # Show some details about the failure
                result = r.get("result", {})
                if "observed_value" in result:
                    print(f"    Observed: {result['observed_value']}")
                if "element_count" in result and "unexpected_count" in result:
                    print(f"    Unexpected count: {result['unexpected_count']}/{result['element_count']}")
        sys.exit(1)
    else:
        print("✅ All checks passed!")

if __name__ == "__main__":
    validate_data("src/data/cleaned/df_limpo.csv")