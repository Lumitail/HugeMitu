from acs.cli.main import main


def test_cli_placeholder(capsys):
    main()
    captured = capsys.readouterr()
    assert "ACS CLI placeholder" in captured.out
