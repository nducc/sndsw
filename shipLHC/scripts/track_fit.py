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

    tracks.clear()
    track_id = 0
    converged_tracks = 0
    good_tracks = 0
    # Open file containing the tracks previously computed by running the tracking
    # change filename and path accordingly
    myFile = ROOT.TFile.Open("/path/to/file.root")
    # Check if the file is open
    if myFile.IsOpen():
        print("ROOT file opened successfully.")
    else:
        print("Failed to open ROOT file.")
    
    tree = myFile.Get("rawConv;1")
    branch = tree.GetBranch("Reco_MuonTracks")
    
    # Loop over the entries in the tree
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        reco_muon_tracks = getattr(tree, "Reco_MuonTracks")

        # Loop over the entries in the branch
        for j in range(reco_muon_tracks.GetEntries()):
            reco_muon_track = reco_muon_tracks.At(j)
            fit_track = reco_muon_track
            if fit_track:
                converged_tracks += 1
                if isGood(fit_track):
                    tracks.push_back(fit_track)
                    track_id += 1
                    good_tracks += 1          
                    if display:
                        display.addEvent(fit_track)
    out_tree.Fill()
    HISTS["track_candidates"].Fill(len(range(reco_muon_tracks.GetEntries())))
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