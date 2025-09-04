
from ..version import main, VERSION

def test_main(capsys):
    msg = f"Version: {VERSION}\n"
    main()
    captured = capsys.readouterr()
    assert captured.out == msg
