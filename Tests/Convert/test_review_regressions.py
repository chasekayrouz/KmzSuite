"""Independent regression reproducers; both fail on the final-r3 merge candidate."""
import subprocess,sys
import numpy as np
from kmzconvert_core import mass_correction as mc

def test_worker_timeout_error_is_propagated():
    code='''
from kmzconvert_core import native
def fail(i): raise TimeoutError("worker failure")
try: list(native.ordered(fail, [0], 1))
except TimeoutError as e:
    assert str(e) == "worker failure"
else: raise AssertionError("failure swallowed")
'''
    process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        try:stdout,stderr=process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            raise AssertionError('ordered() spins on a completed future that raised TimeoutError') from None
        assert process.returncode==0,stderr
    finally:
        if process.poll() is None:process.terminate();process.communicate(timeout=5)

def test_estimator_preserves_float32_scalar_arithmetic():
    mz=np.array([99.998,99.999,100.,100.001,100.002],dtype='f4')
    y=np.array([0.,2.,10.,2.,0.],dtype='f4')
    scans=[mc.ProfileScan(str(i),float(rt),mz,y*scale) for i,(rt,scale) in enumerate([(0.9,1.),(1.,10.),(1.1,1.)])]
    ref=mc.Reference('r',100.,1.);settings=mc.SearchSettings()
    records,_=mc.candidates(ref,scans,settings)
    # local_mass retains the NumPy path for this accepted input, as main did.
    expected=mc.local_mass(scans[1].mz,scans[1].intensity,records[0]['seed_mz'],settings)
    assert records[0]['measurements'][1]==expected

def test_empty_decoded_spectrum_fails_the_tic_guard_not_an_index():
    from kmzconvert_core.binary import FormatError, measure_spectrum
    total, top, mz, valid = measure_spectrum(np.empty(0, np.int64), np.empty(0), .1, 0., 1e-4)
    assert total == 0. and top != top and len(mz) == 0 and not valid
