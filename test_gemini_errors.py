from google.genai import errors


def test_quota_error():
    error = errors.ClientError(
        429,
        {
            "error": {
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    assert error.code == 429

    print("429 error handling test passed")


test_quota_error()