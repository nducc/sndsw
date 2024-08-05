#!/usr/bin/env python3
"""Standalone tracking implementation."""

import argparse
import logging
import numpy as np
from tqdm import tqdm
import ROOT

from shipunit import um, mm

# from pat_rec import Track

import rootUtils as ut


def track_fit(track, fitter, strict=False):
    """Fit a single track candidate in 3d (also works for 2d)."""
    pos = ROOT.TVector3(0, 0, 0.0)
    mom = ROOT.TVector3(0, 0, 100.0)  # default track with high momentum

    # approximate covariance
    sqrt12 = 12**0.5
    res_fine = 122 * um / sqrt12
    res_coarse = 91.5 * mm / sqrt12
    rep = ROOT.genfit.RKTrackRep(13)

    fit_track = ROOT.genfit.Track(rep, pos, mom)

    hit_zs = np.array([hit.z for hit in track.hits])
    hit_xs = np.array([hit.x for hit in track.hits])
    hit_ys = np.array([hit.y for hit in track.hits])
    hit_ids = np.array([hit.hit_id for hit in track.hits], dtype=int)
    det_ids = np.array([hit.det_id for hit in track.hits], dtype=int)
    views = np.array([hit.view for hit in track.hits], dtype=int)

    for i in hit_zs.argsort():
        view = int(views[i])
        hit_covariance = ROOT.TMatrixDSym(2)
        hit_covariance.UnitMatrix()
        hit_covariance[view][view] = res_fine**2
        hit_covariance[(view + 1) % 2][(view + 1) % 2] = res_coarse**2
        hit_coords = ROOT.TVectorD(2)
        hit_coords[0] = hit_xs[i]
        hit_coords[1] = hit_ys[i]

        measurement = ROOT.genfit.PlanarMeasurement(
            hit_coords,
            hit_covariance,
            int(det_ids[i]),
            int(hit_ids[i]),
            ROOT.nullptr,
        )
        measurement.setPlane(
            ROOT.genfit.SharedPlanePtr(
                ROOT.genfit.DetPlane(
                    ROOT.TVector3(0, 0, hit_zs[i]),
                    ROOT.TVector3(1, 0, 0),
                    ROOT.TVector3(0, 1, 0),
                )
            ),
            int(det_ids[i]),
        )

        fit_track.insertPoint(ROOT.genfit.TrackPoint(measurement, fit_track))

    try:
        fit_track.checkConsistency()
    except Exception as e:
        fit_track.Delete()
        raise RuntimeError("Kalman fitter track consistency check failed.") from e

    # do the fit
    fitter.processTrack(fit_track)  # processTrackWithRep(theTrack,rep,True)

    fit_status = fit_track.getFitStatus()
    # fit_track.getFittedState().Print()

    if fit_status.isFitConverged():
        return fit_track
    fit_track.Delete()
    if strict:
        raise RuntimeError("Kalman fit did not converge.")
    return None


def isGood(track):
    """Apply track quality cuts (placeholder)."""
    return True


def main():
    """Fit track candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--inputfile",
        help="""Simulation results to use as input."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--geofile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""File to write the filtered tree to."""
        """Will be recreated if it already exists.""",
    )
    parser.add_argument(
        "-D", "--display", help="Use GenFit event display", action="store_true"
    )
    args = parser.parse_args()
    ROOT.gROOT.SetBatch(not args.display)
    geofile = ROOT.TFile.Open(args.geofile, "read")
    geo = geofile.FAIRGeom  # noqa: F841
    if not args.outputfile:
        args.outputfile = args.inputfile.removesuffix(".root") + "_tracked.root"
    ROOT.gInterpreter.Declare('#include "TGeoMaterialInterface.h"')
    ROOT.gInterpreter.Declare('#include "MaterialEffects.h"')
    ROOT.gInterpreter.Declare('#include "FieldManager.h"')
    ROOT.gInterpreter.Declare('#include "ConstField.h"')
    geo_mat = ROOT.genfit.TGeoMaterialInterface()
    ROOT.genfit.MaterialEffects.getInstance().init(geo_mat)
    bfield = ROOT.genfit.ConstField(0, 0, 0)
    field_manager = ROOT.genfit.FieldManager.getInstance()
    field_manager.init(bfield)
    ROOT.genfit.MaterialEffects.getInstance().setNoEffects()

    kalman_fitter = ROOT.genfit.DAF()
    kalman_fitter.setMaxIterations(50)

    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    tree = inputfile.cbmsim

    outputfile = ROOT.TFile.Open(args.outputfile, "recreate")
    out_tree = tree.CloneTree(0)
    for key in inputfile.GetListOfKeys():
        if key.GetClassName() in ["TH1D", "TH2D"]:
            hist = key.ReadObj()
            hist.Write()

    tracks = ROOT.std.vector("genfit::Track*")()
    out_tree.Branch("genfit_tracks", tracks)
    display = None
    if args.display:
        display = ROOT.genfit.EventDisplay.getInstance()

    # for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
    tracks.clear()
    track_id = 0
    converged_tracks = 0
    good_tracks = 0
    # Loop over the tracks
    myFile = ROOT.TFile.Open("/eos/user/n/nduc/sndlhcsoftware/tracking3D/try_for_genfit_equal_1.root")
    # Check if the file is open
    if myFile.IsOpen():
        print("ROOT file opened successfully.")
    else:
        print("Failed to open ROOT file.")
    
    tree = myFile.Get("rawConv;1")
    branch = tree.GetBranch("Reco_MuonTracks")
    print(f"Branch: {branch}\n")
    
    # Loop over the entries in the tree
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        reco_muon_tracks = getattr(tree, "Reco_MuonTracks")
        print(f"Reco_MuonTrackS: {reco_muon_tracks}\n")

        # Loop over the entries in the branch
        for j in range(reco_muon_tracks.GetEntries()):
            reco_muon_track = reco_muon_tracks.At(j)
            print(f"Reco_MuonTrack: {reco_muon_track}\n")

    # for track_candidate in output_tracks:
    #     hits = []
        # for i in track_candidate:
            # Need to load the track from the outputfile or run_TrackSelection

            # digi_hit = event.Digi_advTargetHits[i]
            # hit = ROOT.Hit()
            # hit.det_id = digi_hit.GetDetectorID()
            # stop = ROOT.TVector3()
            # start = ROOT.TVector3()
            # digi_hit.GetPosition(stop, start)
            # pos = (stop + start) / 2
            # hit.x = pos[0]
            # hit.y = pos[1]
            # hit.z = pos[2]
            # hit.view = int(digi_hit.isVertical())
            # hit.hit_id = i
            # hits.append(hit)
        # track = Track(hits=hits, track_id=track_id)
        # fit_track = track_fit(track, fitter=kalman_fitter)
            fit_track = reco_muon_track
            print(f"Fit track: {fit_track}, Type: {fit_track.ClassName()}\n")
            if fit_track:
                converged_tracks += 1
                if isGood(fit_track):
                    tracks.push_back(fit_track)
                    track_id += 1
                    good_tracks += 1
    print(f"Just before display\n")                
    if display:
        display.addEvent(tracks)
    out_tree.Fill()
    HISTS["track_candidates"].Fill(len(event.track_candidates))
    HISTS["converged_tracks"].Fill(converged_tracks)
    HISTS["good_tracks"].Fill(good_tracks)

    print(f"Just before writing\n")   
    out_tree.Write()
    branch_list = inputfile.BranchList
    branch_list.Add(ROOT.TObjString("genfit_tracks"))
    outputfile.WriteObject(branch_list, "BranchList")
    for key in HISTS:
        HISTS[key].Write()
    outputfile.Write()
    if display:
        display.open()


HISTS = {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ut.bookHist(HISTS, "track_candidates", "", 100, -0.5, 99.5)
    ut.bookHist(HISTS, "converged_tracks", "", 100, -0.5, 99.5)
    ut.bookHist(HISTS, "good_tracks", "", 100, -0.5, -99.5)
    main()