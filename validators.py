"""
Contract validation utilities.
These ensure schema stability across the pipeline.
"""

from schema import CI_REQUIRED_KEYS, CI_SCHEMA_VERSION


def validate_ci_output(ci: dict) -> None:
    """Validate Conversation Intelligence output structure."""

    assert "rules_version" in ci, "Missing rules_version"
    assert ci["rules_version"] == CI_SCHEMA_VERSION, (
        f"Expected rules_version={CI_SCHEMA_VERSION}, "
        f"got {ci['rules_version']}"
    )

    for key, subkeys in CI_REQUIRED_KEYS.items():
        assert key in ci, f"Missing CI key: {key}"

        if subkeys:
            for subkey in subkeys:
                assert subkey in ci[key], f"Missing {key}.{subkey}"
