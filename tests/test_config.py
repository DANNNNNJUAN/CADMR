from cadmr.config import get_bool_env, get_int_env, load_dotenv


def test_load_dotenv_reads_key_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CADMR_TEST_VALUE='hello'\nCADMR_TEST_INT=7\nCADMR_TEST_BOOL=true\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CADMR_TEST_VALUE", raising=False)

    loaded = load_dotenv(env_file)

    assert loaded["CADMR_TEST_VALUE"] == "hello"
    assert get_int_env("CADMR_TEST_INT") == 7
    assert get_bool_env("CADMR_TEST_BOOL") is True
