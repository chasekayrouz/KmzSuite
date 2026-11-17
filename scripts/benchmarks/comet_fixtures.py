"""Reproducible generic LC-MS fixtures. No acquired spectra or personal metadata."""
from dataclasses import dataclass
from pathlib import Path

from lxml import etree
import numpy as np

from Tests.Viewer.mzml_factory import write_mzml


@dataclass(frozen=True)
class Truth:
    id: str
    mz: float
    rt: float
    heights: tuple[float, ...]
    polarity: int = 1
    sigma: float = .035


def write_run(path, truth, *, sample=0, shift=0., mass_ppm=0., profile=False,
              seed=41, noise=20., rt_start=0., rt_stop=5., step=.01,
              gaps=(), rt_unit="minute", scan_windows=True, background=0., spike=False,
              noise_model="uniform"):
    rng = np.random.default_rng(seed)
    scans = []
    noise_draw = (lambda size=None: rng.uniform(0,noise,size)) if noise_model == "uniform" else (lambda size=None: rng.exponential(noise/2,size))
    polarities = sorted({item.polarity for item in truth}) or [1]
    for rt in np.arange(rt_start, rt_stop+step/2, step):
        if any(lo <= rt < hi for lo, hi in gaps):
            continue
        for polarity in polarities:
            masses, values = [], []
            # Equal masses represent chromatographic isomers, not duplicate array entries.
            for mass in sorted({item.mz for item in truth if item.polarity == polarity}):
                signal = sum(item.heights[sample]*np.exp(-.5*((rt-item.rt-shift)/item.sigma)**2)
                             for item in truth if item.mz == mass and item.polarity == polarity)
                center = mass*(1+mass_ppm*1e-6)
                if profile:
                    offsets = np.arange(-12, 13, 2)*1e-6
                    masses.extend(center*(1+offsets))
                    values.extend(signal*np.exp(-.5*(offsets/3e-6)**2)+noise_draw(len(offsets)))
                else:
                    masses.append(center)
                    values.append(signal+noise_draw())
            if background:
                masses.append(1100.)
                values.append(background)
            if spike:
                masses.append(1200.)
                values.append(20000. if abs(rt-2.5)<step/2 else 0.)
            order = np.argsort(masses)
            scans.append(dict(rt=float(rt*60 if rt_unit == "second" else rt), rt_unit=rt_unit,
                              mz=np.asarray(masses)[order], intensity=np.asarray(values)[order],
                              centroid=not profile, polarity=polarity))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mzml(path, scans)
    if scan_windows:
        tree = etree.parse(str(path))
        ns = "{http://psi.hupo.org/ms/mzml}"
        for scan in tree.iter(ns+"scan"):
            window_list = etree.SubElement(scan, ns+"scanWindowList", count="1")
            window = etree.SubElement(window_list, ns+"scanWindow")
            for accession, name, value in (("MS:1000501", "scan window lower limit", "50"),
                                            ("MS:1000500", "scan window upper limit", "1500")):
                etree.SubElement(window, ns+"cvParam", cvRef="MS", accession=accession, name=name,
                                 value=value, unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
    return path


def cohort(root, *, profile=False, seed=41):
    truth = [Truth(f"shared-{i+1:02d}", 150+i*23.17, .8+(i%10)*.34,
                   (12000.+i*300, 7000.+i*200, 600.+i*15, 4200.+i*100)) for i in range(12)]
    truth += [Truth(f"anchor-{i+1:02d}", 500+i*27.13, .5+i*.75,
                    (15000.,13000.,11000.,10000.)) for i in range(6)]
    truth += [Truth("only-first", 720., 1.8, (9000.,0.,0.,0.)),
              Truth("missing-last", 751., 2.4, (10000.,8000.,5000.,0.)),
              Truth("isomer-early", 800., 2.0, (10000.,8000.,7000.,6000.)),
              Truth("isomer-late", 800., 2.4, (8000.,10000.,6000.,7000.)),
              Truth("near-mass-a", 880., 3.3, (11000.,9000.,8000.,7000.)),
              Truth("near-mass-b", 880.*(1+35e-6), 3.3, (7000.,8000.,9000.,10000.)),
              Truth("isotope-retained", 151.003354835, .8, (6000.,3500.,300.,2100.))]
    shifts = (0., .07, -.04, .11)
    paths = [write_run(Path(root)/f"sample-{i+1:02d}.mzML", truth, sample=i, shift=shifts[i],
                       mass_ppm=(0.,2.,-2.,1.)[i], profile=profile, seed=seed+i,
                       background=5000., spike=True) for i in range(4)]
    return paths, truth, shifts
