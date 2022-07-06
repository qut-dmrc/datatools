import pytest

from datatools.runner import only_run, Runner, Delay

class UselessRunner(Runner):
    def __init__(self):
        super(UselessRunner, self).__init__()

    @only_run(seconds=35)
    def do_nothing(self, arg1, arg2, *args, **kwargs):
        return True

@pytest.fixture
def useless_runner():
    runner = UselessRunner()
    yield runner


def test_only_run_ok(useless_runner):
    # it should run the first time
    useless_runner.do_nothing(False, arg1=True, arg2=7)
    return True

def test_only_run_delay(useless_runner):
    try:
        # and fail the second time.
        useless_runner.do_nothing(False, arg1=True, arg2=7)
    except Delay:
        # success, it didn't run
        return True
    return False

def test_only_run_newcmd(useless_runner):
    # but work with different arguments
    useless_runner.do_nothing(True, arg1=True, arg2=7)
    return True

