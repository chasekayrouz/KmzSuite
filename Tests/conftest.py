import gc

import pytest


# Python's cyclic collector runs on whichever thread allocates, so a worker thread could
# destroy GUI-thread Qt objects kept alive by a reference cycle (e.g. a helper QEventLoop
# and its active deadline QTimer pinned by a caught exception's traceback). Qt cannot
# stop such a timer from another thread; it later fires on freed memory. Collect cycles
# only on the main thread, after each test; the young generations then still hold
# everything that test allocated, so this stays cheap.
@pytest.fixture(scope='session', autouse=True)
def collect_cycles_on_main_thread():
    enabled = gc.isenabled()
    gc.disable()
    yield [0]
    gc.collect()  # Final full sweep on the main thread before automatic collection resumes.
    if enabled:
        gc.enable()


@pytest.fixture(autouse=True)
def collect_cycles_after_test(collect_cycles_on_main_thread):
    yield
    # Young generations every test; a full collection periodically so promoted cycles never pile up.
    collect_cycles_on_main_thread[0] += 1
    gc.collect(2 if collect_cycles_on_main_thread[0] % 100 == 0 else 1)


@pytest.fixture
def unqualified():
    """unqualified(*names) makes those dependencies differ from their qualified pins for
    this test, so every exact fast path that checks them takes its original path."""
    from kmzcore import qualified as module
    pins = dict(module.QUALIFIED)

    def unqualify(*names):
        for name in names:
            module.QUALIFIED[name] = 'unqualified'
        module.qualified.cache_clear()

    yield unqualify
    module.QUALIFIED.update(pins)
    module.qualified.cache_clear()
