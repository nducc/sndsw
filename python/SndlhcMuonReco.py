import ROOT
import numpy as np
from array import array
import xml.etree.ElementTree as ET
import shipunit as unit

from hough import hough

def hit_finder(slope, intercept, box_centers, box_ds, tol = 0.) :
    """ Finds hits intersected by Hough line """

    # First check if track at center of box is within box limits
    d = np.abs(box_centers[0,:,1] - (box_centers[0,:,0]*slope + intercept))
    center_in_box = d < (box_ds[0,:,1]+tol)/2.

    # Now check if, assuming line is not in box at box center, the slope is large enough for line to clip the box at corner
    clips_corner = np.abs(slope) > np.abs((d - (box_ds[0,:,1]+tol)/2.)/(box_ds[0,:,0]+tol)/2.)
    
    # If either of these is true, line goes through hit:
    hit_mask = np.logical_or(center_in_box, clips_corner)

    # Return indices
    return np.where(hit_mask)[0]

def numPlanesHit(systems, detector_ids) :
    scifi_stations = []
    mufi_ds_planes = []
    mufi_us_planes = []

    scifi_stations.append( detector_ids[systems == 0]//1000000 )
    mufi_ds_planes.append( (detector_ids[systems == 3]%10000)//1000 )
    mufi_us_planes.append( (detector_ids[systems == 2]%10000)//1000 )

    return len(np.unique(scifi_stations)) + len(np.unique(mufi_ds_planes)) + len(np.unique(mufi_us_planes))

class MuonReco(ROOT.FairTask) :
    " Muon reconstruction "

    def Init(self) :

        self.logger = ROOT.FairLogger.GetLogger()
        if self.logger.IsLogNeeded(ROOT.fair.Severity.info):
           print("Initializing muon reconstruction task!")

        self.lsOfGlobals  = ROOT.gROOT.GetListOfGlobals()
        self.scifiDet = self.lsOfGlobals.FindObject('Scifi')
        self.mufiDet = self.lsOfGlobals.FindObject('MuFilter')
        self.ioman = ROOT.FairRootManager.Instance()
        
        # MC or data - needed for hit timing unit
        if self.ioman.GetInTree().GetName() == 'rawConv': self.isMC = False
        else: self.isMC = True

        # Fetch digi hit collections from online if exist
        sink = self.ioman.GetSink()
        eventTree = None
        if sink:   
            eventTree = sink.GetOutTree()
        if eventTree:
            self.MuFilterHits = eventTree.Digi_MuFilterHits
            self.ScifiHits       = eventTree.Digi_ScifiHits
            self.EventHeader        = eventTree.EventHeader
        else:
        # Try the FairRoot way 
            self.MuFilterHits = self.ioman.GetObject("Digi_MuFilterHits")
            self.ScifiHits = self.ioman.GetObject("Digi_ScifiHits")
            self.EventHeader = self.ioman.GetObject("EventHeader")

        # If that doesn't work, try using standard ROOT
            if self.MuFilterHits == None :
               if self.logger.IsLogNeeded(ROOT.fair.Severity.info):
                  print("Digi_MuFilterHits not in branch list")
               self.MuFilterHits = self.ioman.GetInTree().Digi_MuFilterHits
            if self.ScifiHits == None :
               if self.logger.IsLogNeeded(ROOT.fair.Severity.info):
                  print("Digi_ScifiHits not in branch list")
               self.ScifiHits = self.ioman.GetInTree().Digi_ScifiHits
            if self.EventHeader == None :
               if self.logger.IsLogNeeded(ROOT.fair.Severity.info):
                  print("EventHeader not in branch list")
               self.EventHeader = self.ioman.GetInTree().EventHeader
        
        if self.MuFilterHits == None :
            raise RuntimeError("Digi_MuFilterHits not found in input file.")
        if self.ScifiHits == None :
            raise RuntimeError("Digi_ScifiHits not found in input file.")
        if self.EventHeader == None :
            raise RuntimeError("EventHeader not found in input file.")
        
        # Initialize event counters in case scaling of events is required
        self.scale = 1
        self.events_run = 0
        
        # Initialize hough transform - reading parameter xml file
        tree = ET.parse(self.par_file)
        root = tree.getroot()
        
        # Output track in genfit::Track or sndRecoTrack format
        # Check if genfit::Track format is already forced
        if hasattr(self, "genfitTrack"): pass
        else: self.genfitTrack = int(root[0].text)
        
        self.draw = int(root[1].text)

        track_case_exists = False
        for case in root.findall('tracking_case'):
            if case.get('name') == self.tracking_case:
               track_case_exists = True
               # Use SciFi hits or clusters
               self.Scifi_meas = int(case.find('use_Scifi_clust').text)
               # Maximum absolute value of reconstructed angle (+/- 1 rad is the maximum angle to form a triplet in the SciFi)
               max_angle = float(case.find('max_angle').text)
               
               # Hough space representation
               Hspace_format_exists = False 
               for rep in case.findall('Hough_space_format'):
                   if rep.get('name') == self.Hough_space_format:
                      Hspace_format_exists = True
                      # Number of bins per Hough accumulator axes and range
                      ''' xH and yH are the abscissa and ordinate of the Hough parameter space
                          xz and yz represent horizontal and vertical projections 
                          in the SNDLHC physics coord. system '''
                      n_accumulator_yH = int(rep.find('N_yH_bins').text)
                      yH_min_xz = float(rep.find('yH_min_xz').text)
                      yH_max_xz = float(rep.find('yH_max_xz').text)
                      yH_min_yz = float(rep.find('yH_min_yz').text)
                      yH_max_yz = float(rep.find('yH_max_yz').text)
                      n_accumulator_xH = int(rep.find('N_xH_bins').text)
                      xH_min_xz = float(rep.find('xH_min_xz').text)
                      xH_max_xz = float(rep.find('xH_max_xz').text)
                      xH_min_yz = float(rep.find('xH_min_yz').text)
                      xH_max_yz = float(rep.find('xH_max_yz').text)

                   else: continue
               if not Hspace_format_exists:
                  raise RuntimeError("Unknown Hough space format, check naming in parameter xml file.") 

               # A scale factor for a back-up Hough space having more/less bins than the default one
               # It is useful when fitting some low-E muon tracks, which are curved due to mult. scattering.
               self.HT_space_scale = 1 if case.find('HT_space_scale')==None else float(case.find('HT_space_scale').text)

               # Number of random throws per hit
               self.n_random = int(case.find('n_random').text)
               # MuFilter weight. Muon filter hits are thrown more times than scifi
               self.muon_weight = int(case.find('mufi_weight').text)
               # Minimum number of planes hit in each of the downstream muon filter (if muon filter hits used) or scifi (if muon filter hits not used) views to try to reconstruct a muon
               self.min_planes_hit = int(case.find('min_planes_hit').text)

               # Maximum number of muons to find. To avoid spending too much time on events with lots of downstream activity.
               self.max_reco_muons = int(case.find('max_reco_muons').text)

               # How far away from Hough line hits will be assigned to the muon, for Kalman tracking
               self.tolerance = float(case.find('tolerance').text)

               # Which hits to use for track fitting.
               self.hits_to_fit = case.find('hits_to_fit').text.strip()
               # Which hits to use for triplet condition.
               self.hits_for_triplet = case.find('hits_for_hough').text.strip() if case.find('hits_to_validate')==None else case.find('hits_to_validate').text.strip()
               
               # Detector plane masking. If flag is active, a plane will be masked if its N_hits > Nhits_per_plane.
               # In any case, plane masking will only be applied if solely Scifi hits are used in HT as it is
               # a measure against having many maxima in HT space.
               self.mask_plane = int(case.find('mask_plane').text)
               self.Nhits_per_plane = int(case.find('Nhits_per_plane').text)

               # Enable Gaussian smoothing over the full accumulator space.
               self.smooth_full  = int(case.find('smooth_full').text)
               # Gaussian smoothing parameters. The kernel size is determined as 2*int(truncate*sigma+0.5)+1
               self.sigma = int(case.find('sigma').text)
               self.truncate = int(case.find('truncate').text)
               # Helpers to pick up one of many HT space maxima
               self.n_quantile = float(case.find('n_quantile').text)
               self.res = int(case.find('res').text)

            else: continue
        if not track_case_exists:
           raise RuntimeError("Unknown tracking case, check naming in parameter xml file.")

        # Get speed of light in medium
        self.SpeedLightMedium = self.mufiDet.GetConfParF("MuFilter/DsPropSpeed")

        # Get sensor dimensions from geometry
        self.MuFilter_ds_dx = self.mufiDet.GetConfParF("MuFilter/DownstreamBarY") # Assume y dimensions in vertical bars are the same as x dimensions in horizontal bars.
        self.MuFilter_ds_dy = self.mufiDet.GetConfParF("MuFilter/DownstreamBarY") # Assume y dimensions in vertical bars are the same as x dimensions in horizontal bars.
        self.MuFilter_ds_dz = self.mufiDet.GetConfParF("MuFilter/DownstreamBarZ")

        self.MuFilter_us_dx = self.mufiDet.GetConfParF("MuFilter/UpstreamBarX")
        self.MuFilter_us_dy = self.mufiDet.GetConfParF("MuFilter/UpstreamBarY")
        self.MuFilter_us_dz = self.mufiDet.GetConfParF("MuFilter/UpstreamBarZ")

        self.Scifi_dx = self.scifiDet.GetConfParF("Scifi/channel_width")
        self.Scifi_dy = self.scifiDet.GetConfParF("Scifi/channel_width")
        self.Scifi_dz = self.scifiDet.GetConfParF("Scifi/epoxymat_z") # From Scifi.cxx This is the variable used to define the z dimension of SiPM channels, so seems like the right dimension to use.

        # Get number of readout channels
        self.MuFilter_us_nSiPMs = self.mufiDet.GetConfParI("MuFilter/UpstreamnSiPMs")*self.mufiDet.GetConfParI("MuFilter/UpstreamnSides")
        self.MuFilter_ds_nSiPMs_hor = self.mufiDet.GetConfParI("MuFilter/DownstreamnSiPMs")*self.mufiDet.GetConfParI("MuFilter/DownstreamnSides")
        self.MuFilter_ds_nSiPMs_vert = self.mufiDet.GetConfParI("MuFilter/DownstreamnSiPMs")

        self.Scifi_nPlanes    = self.scifiDet.GetConfParI("Scifi/nscifi")
        self.DS_nPlanes       = self.mufiDet.GetConfParI("MuFilter/NDownstreamPlanes")
        self.max_n_hits_plane = 3
        self.max_n_Scifi_hits = self.max_n_hits_plane*2*self.Scifi_nPlanes
        self.max_n_DS_hits    = self.max_n_hits_plane*(2*self.DS_nPlanes-1)

        # get the distance between 1st and last detector planes to be used in the track fit.
        # a z_offset is used to shift detector hits so to have smaller Hough parameter space
        # Using geometers measurements! For safety, add a 5-cm-buffer in detector lengths and a 2.5-cm one to z_offset.
        # This is done to account for possible det. position shifts/mismatches going from geom. measurements and sndsw physics CS.
        if self.hits_for_triplet.find('sf') >= 0 and self.hits_for_triplet.find('ds') >= 0:
           det_Zlen = (self.mufiDet.GetConfParF("MuFilter/Muon9Dy") - self.scifiDet.GetConfParF("Scifi/Ypos0"))*unit.cm + 5.0*unit.cm
           z_offset = self.scifiDet.GetConfParF("Scifi/Ypos0")*unit.cm - 2.5*unit.cm
        elif self.hits_for_triplet == 'sf':
           det_Zlen = (self.scifiDet.GetConfParF("Scifi/Ypos4") - self.scifiDet.GetConfParF("Scifi/Ypos0"))*unit.cm + 5.0*unit.cm
           z_offset = self.scifiDet.GetConfParF("Scifi/Ypos0")*unit.cm - 2.5*unit.cm
        elif self.hits_for_triplet == 'ds':
           det_Zlen = (self.mufiDet.GetConfParF("MuFilter/Muon9Dy") - self.mufiDet.GetConfParF("MuFilter/Muon6Dy"))*unit.cm + 5.0*unit.cm
           z_offset = self.mufiDet.GetConfParF("MuFilter/Muon6Dy")*unit.cm - 2.5*unit.cm
        # this use case is not tested with an z offset yet
        if self.tracking_case.find('nu_') >= 0: z_offset = 0*unit.cm 
        #other use cases come here if ever added

        # Initialize Hough transforms for both views:
        if self.Hough_space_format == 'normal':
            # rho-theta representation - must exclude theta range for tracks passing < 3 det. planes
            self.h_ZX = hough(n_accumulator_yH, [yH_min_xz, yH_max_xz], n_accumulator_xH, [-max_angle+np.pi/2., max_angle+np.pi/2.], z_offset, self.Hough_space_format, self.HT_space_scale, det_Zlen)
            self.h_ZY = hough(n_accumulator_yH, [yH_min_yz, yH_max_yz], n_accumulator_xH, [-max_angle+np.pi/2., max_angle+np.pi/2.], z_offset, self.Hough_space_format, self.HT_space_scale, det_Zlen)
        else:
            self.h_ZX = hough(n_accumulator_yH, [yH_min_xz, yH_max_xz], n_accumulator_xH, [xH_min_xz, xH_max_xz], z_offset, self.Hough_space_format, self.HT_space_scale, det_Zlen)
            self.h_ZY = hough(n_accumulator_yH, [yH_min_yz, yH_max_yz], n_accumulator_xH, [xH_min_yz, xH_max_yz], z_offset, self.Hough_space_format, self.HT_space_scale, det_Zlen)

        self.h_ZX.smooth_full = self.smooth_full
        self.h_ZY.smooth_full = self.smooth_full
        self.h_ZX.sigma = self.sigma
        self.h_ZX.truncate = self.truncate
        self.h_ZY.sigma = self.sigma
        self.h_ZY.truncate = self.truncate

        self.h_ZX.n_quantile = self.n_quantile
        self.h_ZX.res = self.res
        self.h_ZY.n_quantile = self.n_quantile
        self.h_ZY.res = self.res

        if self.hits_to_fit == "sf" : self.track_type = 11
        elif self.hits_to_fit == "ds": self.track_type = 13
        else : self.track_type = 15
        
        # To keep temporary detector information
        self.a = ROOT.TVector3()
        self.b = ROOT.TVector3()

        # genfitTrack variable setting by hand DEBUG
        # Needs to be 1 when producing root files for track_fit.py and 2dEventDisplay.py
        # Make the value agree with the one in Monitor.py when running run_TrackSelections.py
        # self.genfitTrack  whould always be in agreement with the genfitTrack variable of the parameter file
        self.genfitTrack = 1

        # check if track container exists
        if self.ioman.GetObject('Reco_MuonTracks') != None:
             print(f"First\n")
             self.kalman_tracks = self.ioman.GetObject('Reco_MuonTracks')
             if self.logger.IsLogNeeded(ROOT.fair.Severity.info):
                print('Branch activated by another task!')

        else:
        #   Now initialize output in genfit::track or sndRecoTrack format
           if self.genfitTrack:
                self.kalman_tracks = ROOT.std.vector("genfit::Track*")()
                if hasattr(self, "standalone") and self.standalone:
                    self.ioman.Register("Reco_MuonTracks", "", self.kalman_tracks, ROOT.kTRUE)

           else:
              self.kalman_tracks = ROOT.TClonesArray("sndRecoTrack")
              if hasattr(self, "standalone") and self.standalone:
                 self.ioman.Register("Reco_MuonTracks", "", self.kalman_tracks, ROOT.kTRUE)

        # initialize detector class with EventHeader(runN), if SNDLHCEventHeader detected
        # only needed if using HT tracking manager, i.e. standalone
        if self.EventHeader.IsA().GetName()=='SNDLHCEventHeader' and hasattr(self, "standalone") and self.standalone and not self.isMC :
           self.scifiDet.InitEvent(self.EventHeader)
           self.mufiDet.InitEvent(self.EventHeader)

        # internal storage of clusters
        if self.Scifi_meas: 
            self.clusScifi = ROOT.TObjArray(100)
        
        # Kalman filter stuff

        geoMat = ROOT.genfit.TGeoMaterialInterface()
        bfield = ROOT.genfit.ConstField(0, 0, 0)
        fM = ROOT.genfit.FieldManager.getInstance()
        fM.init(bfield)
        ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
        ROOT.genfit.MaterialEffects.getInstance().setNoEffects()
        
        self.kalman_fitter = ROOT.genfit.KalmanFitter()
        self.kalman_fitter.setMaxIterations(50)
        self.kalman_sigmaScifi_spatial = self.Scifi_dx / 12**0.5
        self.kalman_sigmaMufiUS_spatial = self.MuFilter_us_dy / 12**0.5
        # For DS 3D tracking: take plane resolution to be the maximum between x and y:
        # Resolution along y
        self.ds_res_y = self.MuFilter_ds_dy/ 12**0.5
        # Compute resolution along x from ToF with error propagation
        delta_t = 150*1e-3/np.sqrt(2)
        delta_calibration = 1/12**0.5
        self.ds_res_x = np.sqrt((0.5*self.SpeedLightMedium*2*delta_t)**2.0 + delta_calibration**2.0)
        self.kalman_sigmaMufiDS_spatial = max(self.ds_res_y, self.ds_res_x)
        print(f"Resolution DS kalman: {self.kalman_sigmaMufiDS_spatial}")  # DEBUG

        # Save the HT line prediction in a ntuple
        self.ntuple = ROOT.TNtuple("HT_prediction","HT_prediction","eventN:zy_a:zy_b:zx_a:zx_b")

        # Init() MUST return int
        return 0
    
    def SetScaleFactor(self, scale):
        self.scale = scale

    def SetParFile(self, file_name):
        self.par_file = file_name
    
    def SetTrackingCase(self, case):
        self.tracking_case = case

    def SetHoughSpaceFormat(self, Hspace_format):
        self.Hough_space_format = Hspace_format

    def ForceGenfitTrackFormat(self):
        self.genfitTrack = 1

    # flag showing the task is run seperately from other tracking tasks
    def SetStandalone(self):
        self.standalone = 1

    def Exec(self, opt) :
        if self.genfitTrack:
            self.kalman_tracks.clear()
        else:
            self.kalman_tracks.Clear('C')

        # Set scaling in case task is run seperately from other tracking tasks
        if self.scale>1 and self.standalone:
           if ROOT.gRandom.Rndm() > 1.0/self.scale: return

        self.events_run += 1
        # print(f"Event number: {self.events_run}") # DEBUG
        hit_collection = {"pos" : [[], [], []],
                          "d" : [[], [], []],
                          "vert" : [],
                          "index" : [],
                          "system" : [],
                          "detectorID" : [],
                          "B" : [[], [], []],
                          "time": [],
                          "mask": [],
                          "hitid": []}  # hitid: integer number different for every hit

        hit_id = 0                  

        if ("us" in self.hits_to_fit) or ("ds" in self.hits_to_fit) or ("ve" in self.hits_to_fit) :
            # Loop through muon filter hits
            for i_hit, muFilterHit in enumerate(self.MuFilterHits) :
                # Don't use veto for fitting
                if muFilterHit.GetSystem() == 1 :
                    if "ve" not in self.hits_to_fit :
                        continue
                elif muFilterHit.GetSystem() == 2 :
                    if "us" not in self.hits_to_fit :
                        continue
                elif muFilterHit.GetSystem() == 3 :
                    if "ds" not in self.hits_to_fit :
                        continue
                    # keep only horizontal counts for ds 3D tracking
                    if muFilterHit.isVertical() :
                        continue
                else :
                    if self.logger.IsLogNeeded(ROOT.fair.Severity.warn):
                       print("WARNING! Unknown MuFilter system!!")

                self.mufiDet.GetPosition(muFilterHit.GetDetectorID(), self.a, self.b)

                if muFilterHit.GetSystem() == 3 :   # Get x position from ToF for 3D DS tracking
                    # Length of the horizontal detector bar
                    L = abs(self.b.X()-self.a.X())
                    # Compute time difference between left (0) and right (1) readouts
                    t0 = self.mufiDet.GetCorrectedTime(muFilterHit.GetDetectorID(), 0, muFilterHit.GetTime(0)*6.25, 0)
                    t1 = self.mufiDet.GetCorrectedTime(muFilterHit.GetDetectorID(), 1, muFilterHit.GetTime(1)*6.25, 0)
                    DeltaT = t0-t1
                    dummy = -999.0
                    if np.isclose(DeltaT, dummy):
                        continue
                    # Distance between left readout and hit along x    
                    x0 = 0.5*(L+DeltaT*self.SpeedLightMedium)
                    # x position of the horizontal hit
                    x = self.a.X() - x0
                    hit_collection["pos"][0].append(x)

                else :
                    hit_collection["pos"][0].append(self.a.X())

                hit_collection["pos"][1].append(self.a.Y())
                hit_collection["pos"][2].append(self.a.Z())

                hit_collection["B"][0].append(self.b.X())
                hit_collection["B"][1].append(self.b.Y())
                hit_collection["B"][2].append(self.b.Z())

                hit_collection["vert"].append(muFilterHit.isVertical())
                hit_collection["system"].append(muFilterHit.GetSystem())

                hit_collection["d"][0].append(self.MuFilter_ds_dx)
                hit_collection["d"][2].append(self.MuFilter_ds_dz)

                hit_collection["index"].append(i_hit)
                
                hit_collection["detectorID"].append(muFilterHit.GetDetectorID())
                hit_collection["mask"].append(False)

                hit_collection["hitid"].append(hit_id)
                hit_id+=1

                times = []
                # Downstream
                if muFilterHit.GetSystem() == 3 :
                    hit_collection["d"][1].append(self.MuFilter_ds_dx)
                    for ch in range(self.MuFilter_ds_nSiPMs_hor):
                        if muFilterHit.isVertical() and ch==self.MuFilter_ds_nSiPMs_vert: break
                        if self.isMC: 
                          times.append(muFilterHit.GetAllTimes()[ch]) #already in ns
                        else: 
                          times.append(muFilterHit.GetAllTimes()[ch]*6.25) #tdc2ns
                # Upstream
                else :
                    hit_collection["d"][1].append(self.MuFilter_us_dy)
                    for ch in range(self.MuFilter_us_nSiPMs):
                        if self.isMC:
                           times.append(muFilterHit.GetAllTimes()[ch]) #already in ns
                        else: times.append(muFilterHit.GetAllTimes()[ch]*6.25) #tdc2ns
                hit_collection["time"].append(times)

        if "sf" in self.hits_to_fit :
            if self.Scifi_meas:
               # Make scifi clusters
               self.clusScifi.Clear()
               self.scifiCluster()

               # Loop through scifi clusters
               for i_clust, scifiCl in enumerate(self.clusScifi) :
                   scifiCl.GetPosition(self.a,self.b)

                   hit_collection["pos"][0].append(self.a.X())
                   hit_collection["pos"][1].append(self.a.Y())
                   hit_collection["pos"][2].append(self.a.Z())

                   hit_collection["B"][0].append(self.b.X())
                   hit_collection["B"][1].append(self.b.Y())
                   hit_collection["B"][2].append(self.b.Z())

                   # take the cluster size as the active area size
                   hit_collection["d"][0].append(scifiCl.GetN()*self.Scifi_dx)
                   hit_collection["d"][1].append(scifiCl.GetN()*self.Scifi_dy)
                   hit_collection["d"][2].append(self.Scifi_dz)

                   if int(scifiCl.GetFirst()/100000)%10==1:
                      hit_collection["vert"].append(True)
                   else: hit_collection["vert"].append(False)
                   hit_collection["index"].append(i_clust)

                   hit_collection["system"].append(0)
                   hit_collection["detectorID"].append(scifiCl.GetFirst())
                   hit_collection["mask"].append(False)

                   hit_collection["hitid"].append(hit_id)
                   hit_id+=1

                   times = []
                   if self.isMC : times.append(scifiCl.GetTime()/6.25) # for MC, hit time is in ns. Then for MC Scifi cluster time one has to divide by tdc2ns
                   else: times.append(scifiCl.GetTime()) # already in ns
                   hit_collection["time"].append(times)

            else:
                 if self.hits_for_triplet == 'sf' and self.hits_to_fit == 'sf':
                   # Loop through scifi hits and count hits per projection and plane
                   N_plane_ZY = {1:0, 2:0, 3:0, 4:0, 5:0}
                   N_plane_ZX = {1:0, 2:0, 3:0, 4:0, 5:0}
                   for scifiHit in self.ScifiHits:
                      if not scifiHit.isValid(): continue
                      if scifiHit.isVertical(): 
                         N_plane_ZX[scifiHit.GetStation()] += 1
                      else:
                         N_plane_ZY[scifiHit.GetStation()] += 1
                   if self.mask_plane:
                      mask_plane_ZY = []
                      mask_plane_ZX = []
                      # sorting
                      N_plane_ZY = dict(sorted(N_plane_ZY.items(), key=lambda item: item[1], reverse = True))
                      N_plane_ZX = dict(sorted(N_plane_ZX.items(), key=lambda item: item[1], reverse = True))
                      # count planes with hits
                      n_zx = self.Scifi_nPlanes - list(N_plane_ZX.values()).count(0)
                      n_zy = self.Scifi_nPlanes - list(N_plane_ZY.values()).count(0)
                      # check with min number of hit planes
                      if n_zx < self.min_planes_hit or n_zy < self.min_planes_hit: return
                      # mask busiest planes until there are at least 3 planes with hits left
                      for ii in range(n_zx-self.min_planes_hit):
                          if list(N_plane_ZX.values())[ii] > self.Nhits_per_plane:
                             mask_plane_ZX.append(list(N_plane_ZX.keys())[ii])
                      for ii in range(n_zy-self.min_planes_hit):
                          if list(N_plane_ZY.values())[ii] > self.Nhits_per_plane:
                             mask_plane_ZY.append(list(N_plane_ZY.keys())[ii])

                 # Loop through scifi hits
                 for i_hit, scifiHit in enumerate(self.ScifiHits) :
                     if not scifiHit.isValid(): continue 
                     self.scifiDet.GetSiPMPosition(scifiHit.GetDetectorID(), self.a, self.b)
                     hit_collection["pos"][0].append(self.a.X())
                     hit_collection["pos"][1].append(self.a.Y())
                     hit_collection["pos"][2].append(self.a.Z())

                     hit_collection["B"][0].append(self.b.X())
                     hit_collection["B"][1].append(self.b.Y())
                     hit_collection["B"][2].append(self.b.Z())

                     hit_collection["d"][0].append(self.Scifi_dx)
                     hit_collection["d"][1].append(self.Scifi_dy)
                     hit_collection["d"][2].append(self.Scifi_dz)

                     hit_collection["vert"].append(scifiHit.isVertical())
                     hit_collection["index"].append(i_hit)

                     hit_collection["system"].append(0)

                     hit_collection["detectorID"].append(scifiHit.GetDetectorID())
                     
                     if self.hits_for_triplet == 'sf' and self.hits_to_fit == 'sf' and self.mask_plane:
                       if (scifiHit.isVertical()==0 and scifiHit.GetStation() in mask_plane_ZY) or (scifiHit.isVertical() and scifiHit.GetStation() in mask_plane_ZX):
                          hit_collection["mask"].append(True)
                       else: hit_collection["mask"].append(False)
                     else:
                          hit_collection["mask"].append(False)

                     hit_collection["hitid"].append(hit_id)
                     hit_id+=1

                     times = []

                     if self.isMC : times.append(scifiHit.GetTime()) # already in ns
                     else: times.append(scifiHit.GetTime()*6.25) #tdc2ns
                     hit_collection["time"].append(times)

        # If no hits, return
        if len(hit_collection['pos'][0])==0: return

        # Make the hit collection numpy arrays.
        for key, item in hit_collection.items() :
            if key == 'vert' :
                this_dtype = np.bool_
            elif key == 'mask' :
                this_dtype = np.bool_
            elif key == "index" or key == "system" or key == "detectorID" or key== "hitid":
                this_dtype = np.int32
            elif key != 'time' :
                this_dtype = np.float32
            if key== 'time':
               length = max(map(len, item))
               hit_collection[key] = np.stack(np.array([xi+[None]*(length-len(xi)) for xi in item]), axis = 1)
            else: hit_collection[key] = np.array(item, dtype = this_dtype)

        # Useful for later
        triplet_condition_system = []
        if "sf" in self.hits_for_triplet :
            triplet_condition_system.append(0)
        if "ve" in self.hits_for_triplet :
            triplet_condition_system.append(1)
        if "us" in self.hits_for_triplet :
            triplet_condition_system.append(2)
        if "ds" in self.hits_for_triplet :
            triplet_condition_system.append(3)

        # Reconstruct muons until there are not enough hits in downstream muon filter
        for i_muon in range(self.max_reco_muons) :

            # For 3D DS tracking, consider horizontal hits only both in triplet_hits_horizontal and triplet_hits_vertical
            triplet_hits_horizontal = np.array([np.isin(hit_collection["system"][i], triplet_condition_system) if hit_collection["system"][i]==3 else
                                                np.logical_and(~hit_collection["vert"][i], np.isin(hit_collection["system"][i], triplet_condition_system)) for i in range(len(hit_collection["detectorID"]))])

            triplet_hits_ds_horizontal = np.logical_and(~hit_collection["vert"], np.logical_and(np.isin(hit_collection["system"], triplet_condition_system), hit_collection["system"]==3))
            
            triplet_hits_vertical = np.array([np.isin(hit_collection["system"][i], triplet_condition_system) if hit_collection["system"][i]==3 else
                                                np.logical_and(hit_collection["vert"][i], np.isin(hit_collection["system"][i], triplet_condition_system)) for i in range(len(hit_collection["detectorID"]))])

            # print(triplet_hits_horizontal)    # DEBUG

            # # prevent to continue if triplet_hits_horizontal is empty
            # if len(triplet_hits_horizontal)<=0: break

            n_planes_ZY = numPlanesHit(hit_collection["system"][triplet_hits_horizontal],
                                       hit_collection["detectorID"][triplet_hits_horizontal])
            n_planes_ds_ZY = numPlanesHit(hit_collection["system"][triplet_hits_ds_horizontal],
                                       hit_collection["detectorID"][triplet_hits_ds_horizontal])

            if n_planes_ZY < self.min_planes_hit or n_planes_ds_ZY < self.min_planes_hit:
                break

            n_planes_ZX = numPlanesHit(hit_collection["system"][triplet_hits_vertical],
                                       hit_collection["detectorID"][triplet_hits_vertical])
            if n_planes_ZX < self.min_planes_hit :
                break

            # Get hits in hough transform format
            muon_hits_horizontal = np.logical_and(~hit_collection["mask"],
                                                   np.isin(hit_collection["system"], [1, 2, 3]))
            muon_hits_vertical = np.logical_and(~hit_collection["mask"],
                                                 np.isin(hit_collection["system"], [1, 2, 3]))
            scifi_hits_horizontal = np.logical_and( np.logical_and( ~hit_collection["vert"], ~hit_collection["mask"]),
                                                    np.isin(hit_collection["system"], [0]))
            scifi_hits_vertical = np.logical_and( np.logical_and( hit_collection["vert"], ~hit_collection["mask"]),
                                                  np.isin(hit_collection["system"], [0]))


            ZY = np.dstack([np.concatenate([np.tile(hit_collection["pos"][2][muon_hits_horizontal], self.muon_weight),
                                            hit_collection["pos"][2][scifi_hits_horizontal]]),
                            np.concatenate([np.tile(hit_collection["pos"][1][muon_hits_horizontal], self.muon_weight),
                                            hit_collection["pos"][1][scifi_hits_horizontal]])])[0]

            d_ZY = np.dstack([np.concatenate([np.tile(hit_collection["d"][2][muon_hits_horizontal], self.muon_weight),
                                              hit_collection["d"][2][scifi_hits_horizontal]]),
                              np.concatenate([np.tile(hit_collection["d"][1][muon_hits_horizontal], self.muon_weight),
                                              hit_collection["d"][1][scifi_hits_horizontal]])])[0]

            ZX = np.dstack([np.concatenate([np.tile(hit_collection["pos"][2][muon_hits_vertical], self.muon_weight),
                                            hit_collection["pos"][2][scifi_hits_vertical]]),
                            np.concatenate([np.tile(hit_collection["pos"][0][muon_hits_vertical], self.muon_weight),
                                            hit_collection["pos"][0][scifi_hits_vertical]])])[0]

            d_ZX = np.dstack([np.concatenate([np.tile(hit_collection["d"][2][muon_hits_vertical], self.muon_weight),
                                              hit_collection["d"][2][scifi_hits_vertical]]),
                              np.concatenate([np.tile(hit_collection["d"][0][muon_hits_vertical], self.muon_weight),
                                              hit_collection["d"][0][scifi_hits_vertical]])])[0]

            is_scaled = False
            ZY_hough = self.h_ZY.fit_randomize(ZY, d_ZY, self.n_random, is_scaled, self.draw)
            ZX_hough = self.h_ZX.fit_randomize(ZX, d_ZX, self.n_random, is_scaled, self.draw)

            tol = self.tolerance
            # Special treatment for events with low hit occupancy - increase tolerance
            # For Scifi-only tracks
            if len(hit_collection["detectorID"]) <= self.max_n_Scifi_hits  and self.hits_for_triplet == 'sf' and self.hits_to_fit == 'sf' :
               # as there are masked Scifi planes, make sure to use hit counts before the masking
               if max(N_plane_ZX.values()) <= self.max_n_hits_plane  and max(N_plane_ZY.values()) <= self.max_n_hits_plane :
                  tol = 5*self.tolerance
            # for DS-only tracks#
            if self.hits_for_triplet == 'ds' and self.hits_to_fit == 'ds' :
               # Loop through hits and count hits per projection and plane
               N_plane_ZY = {0:0, 1:0, 2:0, 3:0}
               N_plane_ZX = {0:0, 1:0, 2:0, 3:0}
               for item in range(len(hit_collection["detectorID"])):
                   if hit_collection["vert"][item]: N_plane_ZX[(hit_collection["detectorID"][item]%10000)//1000] += 1
                   else: N_plane_ZY[(hit_collection["detectorID"][item]%10000)//1000] += 1
               if max(N_plane_ZX.values()) <= self.max_n_hits_plane and max(N_plane_ZY.values()) <= self.max_n_hits_plane :
                  tol = 3*self.tolerance

            track_hits_for_triplet_ZY = hit_finder(ZY_hough[0], ZY_hough[1], 
                                                   np.dstack([hit_collection["pos"][2][triplet_hits_horizontal],
                                                              hit_collection["pos"][1][triplet_hits_horizontal]]),
                                                   np.dstack([hit_collection["d"][2][triplet_hits_horizontal],
                                                              hit_collection["d"][1][triplet_hits_horizontal]]), tol)

            track_hits_for_triplet_ZX = hit_finder(ZX_hough[0], ZX_hough[1], 
                                                   np.dstack([hit_collection["pos"][2][triplet_hits_vertical],
                                                              hit_collection["pos"][0][triplet_hits_vertical]]),
                                                   np.dstack([hit_collection["d"][2][triplet_hits_vertical],
                                                              hit_collection["d"][0][triplet_hits_vertical]]), tol)
                                                   
            n_planes_hit_ZY = numPlanesHit(hit_collection["system"][triplet_hits_horizontal][track_hits_for_triplet_ZY],
                                           hit_collection["detectorID"][triplet_hits_horizontal][track_hits_for_triplet_ZY])

            n_planes_hit_ZX = numPlanesHit(hit_collection["system"][triplet_hits_vertical][track_hits_for_triplet_ZX],
                                           hit_collection["detectorID"][triplet_hits_vertical][track_hits_for_triplet_ZX])

            # For failed SciFi track fits, in events with little hit activity, try using less Hough-space bins
            if (self.hits_to_fit == 'sf' and len(hit_collection["detectorID"]) <= self.max_n_Scifi_hits and \
                (n_planes_hit_ZY < self.min_planes_hit or n_planes_hit_ZX < self.min_planes_hit)):

                is_scaled = True
                ZY_hough = self.h_ZY.fit_randomize(ZY, d_ZY, self.n_random, is_scaled, self.draw)
                ZX_hough = self.h_ZX.fit_randomize(ZX, d_ZX, self.n_random, is_scaled, self.draw)

                # Check if track intersects minimum number of hits in each plane.
                track_hits_for_triplet_ZY = hit_finder(ZY_hough[0], ZY_hough[1],
                                                np.dstack([hit_collection["pos"][2][triplet_hits_horizontal],
                                                           hit_collection["pos"][1][triplet_hits_horizontal]]),
                                                np.dstack([hit_collection["d"][2][triplet_hits_horizontal],
                                                           hit_collection["d"][1][triplet_hits_horizontal]]), tol)

                n_planes_hit_ZY = numPlanesHit(hit_collection["system"][triplet_hits_horizontal][track_hits_for_triplet_ZY],
                                               hit_collection["detectorID"][triplet_hits_horizontal][track_hits_for_triplet_ZY])
                if n_planes_hit_ZY < self.min_planes_hit: 
                   break

                track_hits_for_triplet_ZX = hit_finder(ZX_hough[0], ZX_hough[1],
                                                np.dstack([hit_collection["pos"][2][triplet_hits_vertical],
                                                           hit_collection["pos"][0][triplet_hits_vertical]]),
                                                np.dstack([hit_collection["d"][2][triplet_hits_vertical],
                                                           hit_collection["d"][0][triplet_hits_vertical]]), tol)
                n_planes_hit_ZX = numPlanesHit(hit_collection["system"][triplet_hits_vertical][track_hits_for_triplet_ZX],
                                               hit_collection["detectorID"][triplet_hits_vertical][track_hits_for_triplet_ZX])

            if n_planes_hit_ZY < self.min_planes_hit or n_planes_hit_ZX < self.min_planes_hit: 
               break

#                print("Found {0} downstream ZX planes associated to muon track".format(n_planes_ds_ZX))
#                print("Found {0} downstream ZY planes associated to muon track".format(n_planes_ds_ZY))
            
            # This time with all the hits, not just triplet condition.
            
            # For 3D DS tracking: for a hit to be recorded in the DS is a sufficient condition to be considered in both vertical and horizontal planes
            horizontal_condition = ~hit_collection["vert"] | hit_collection["system"]==3
            vertical_condition = hit_collection["vert"] | hit_collection["system"]==3

            # hough slope and intercept
            print(f"Horizontal Hough slope: {ZY_hough[0]}, intercept: {ZY_hough[1]}\n")
            
            track_hits_ZY = hit_finder(ZY_hough[0], ZY_hough[1], 
                                       np.dstack([hit_collection["pos"][2][horizontal_condition], 
                                                  hit_collection["pos"][1][horizontal_condition]]), 
                                       np.dstack([hit_collection["d"][2][vertical_condition],
                                                  hit_collection["d"][1][vertical_condition]]), tol)

            track_hits_ZX = hit_finder(ZX_hough[0], ZX_hough[1], 
                                       np.dstack([hit_collection["pos"][2][vertical_condition], 
                                                  hit_collection["pos"][0][vertical_condition]]), 
                                       np.dstack([hit_collection["d"][2][vertical_condition], 
                                                  hit_collection["d"][0][vertical_condition]]), tol)

            # save the HT line prediction in a Ntuple
            self.ntuple.Fill(self.EventHeader.GetEventNumber(), ZY_hough[0], ZY_hough[1], ZX_hough[0], ZX_hough[1])     

            # Onto Kalman fitter (based on SndlhcTracking.py)
            posM    = ROOT.TVector3(0, 0, 0.)
            momM = ROOT.TVector3(0,0,100.)  # default track with high momentum

            # approximate covariance
            covM = ROOT.TMatrixDSym(6)
            if self.hits_to_fit.find('sf') >= 0:
                res = self.kalman_sigmaScifi_spatial
            if self.hits_to_fit == 'ds':
                res = self.kalman_sigmaMufiDS_spatial
            for  i in range(3):   covM[i][i] = res*res
            for  i in range(3,6): covM[i][i] = ROOT.TMath.Power(res / (4.*2.) / ROOT.TMath.Sqrt(3), 2)
            rep = ROOT.genfit.RKTrackRep(13)

            # start state
            state = ROOT.genfit.MeasuredStateOnPlane(rep)
            rep.setPosMomCov(state, posM, momM, covM)

            # create track
            seedState = ROOT.TVectorD(6)
            seedCov   = ROOT.TMatrixDSym(6)
            rep.get6DStateCov(state, seedState, seedCov)

            theTrack = ROOT.genfit.Track(rep, seedState, seedCov)

            # Sort measurements in Z

            hit_z = np.concatenate([hit_collection["pos"][2][vertical_condition][track_hits_ZX],
                                    hit_collection["pos"][2][horizontal_condition][track_hits_ZY]])

            hit_A0 = np.concatenate([hit_collection["pos"][0][vertical_condition][track_hits_ZX],
                                     hit_collection["pos"][0][horizontal_condition][track_hits_ZY]])

            hit_A1 = np.concatenate([hit_collection["pos"][1][vertical_condition][track_hits_ZX],
                                     hit_collection["pos"][1][horizontal_condition][track_hits_ZY]])
            
            hit_B0 = np.concatenate([hit_collection["B"][0][vertical_condition][track_hits_ZX],
                                     hit_collection["B"][0][horizontal_condition][track_hits_ZY]])

            hit_B1 = np.concatenate([hit_collection["B"][1][vertical_condition][track_hits_ZX],
                                     hit_collection["B"][1][horizontal_condition][track_hits_ZY]])

            hit_B2 = np.concatenate([hit_collection["B"][2][vertical_condition][track_hits_ZX],
                                     hit_collection["B"][2][horizontal_condition][track_hits_ZY]])

            hit_detid = np.concatenate([hit_collection["detectorID"][vertical_condition][track_hits_ZX],
                                        hit_collection["detectorID"][horizontal_condition][track_hits_ZY]])

            hit_ids = np.concatenate([hit_collection["hitid"][vertical_condition][track_hits_ZX],
                                        hit_collection["hitid"][horizontal_condition][track_hits_ZY]])

            kalman_spatial_sigma = np.concatenate([hit_collection["d"][0][vertical_condition][track_hits_ZX] / 12**0.5,
                                                   hit_collection["d"][1][horizontal_condition][track_hits_ZY] / 12**0.5])

            # Maximum distance. Use (d_xy/2**2 + d_z/2**2)**0.5
            kalman_max_dis = np.concatenate([((hit_collection["d"][0][vertical_condition][track_hits_ZX]/2.)**2 +
                                              (hit_collection["d"][2][vertical_condition][track_hits_ZX]/2.)**2)**0.5,
                                             ((hit_collection["d"][1][horizontal_condition][track_hits_ZY]/2.)**2 +
                                              (hit_collection["d"][2][horizontal_condition][track_hits_ZY]/2.)**2)**0.5])

            hitID = 0 # Does it matter? We don't have a global hit ID.

            hit_time = {}
            for ch in range(hit_collection["time"].shape[0]):
                hit_time[ch] = np.concatenate([hit_collection["time"][ch][vertical_condition][track_hits_ZX],
                                      hit_collection["time"][ch][horizontal_condition][track_hits_ZY]])


            measurement_counter = 0 # DEBUG
            for i_z_sorted in hit_z.argsort() :
                measurement_counter+=1  # DEBUG
                tp = ROOT.genfit.TrackPoint()
                hitCov = ROOT.TMatrixDSym(2)
                hitCov.UnitMatrix()
                hitCov[0][0] = self.ds_res_x**2
                # print(f"hitCov[0][0]: {hitCov[0][0]}")
                hitCov[1][1] = self.ds_res_y**2
                # print(f"hitCov[1][1]: {hitCov[1][1]}")

                hit_coords = ROOT.TVectorD(2)
                hit_coords[0] = hit_A0[i_z_sorted]
                hit_coords[1] = hit_A1[i_z_sorted]
                
                # measurement = ROOT.genfit.WireMeasurement(ROOT.TVectorD(7, array('d', [hit_A0[i_z_sorted],
                #                                                                        hit_A1[i_z_sorted],
                #                                                                        hit_z[i_z_sorted],
                #                                                                        hit_B0[i_z_sorted],
                #                                                                        hit_B1[i_z_sorted],
                #                                                                        hit_B2[i_z_sorted],
                #                                                                        0.])),
                #                                           hitCov,
                #                                           1, # detid?
                #                                           6, # hitid?
                #                                           tp)

                measurement = ROOT.genfit.PlanarMeasurement(
                    hit_coords,
                    hitCov,
                    int(hit_detid[i_z_sorted]),
                    int(hit_ids[i_z_sorted]),
                    ROOT.nullptr,
                )

                print(f"measurement_counter: {measurement_counter}") # DEBUG

                measurement.setPlane(
                            ROOT.genfit.SharedPlanePtr(
                                ROOT.genfit.DetPlane(
                                    ROOT.TVector3(0, 0, hit_z[i_z_sorted]),
                                    ROOT.TVector3(1, 0, 0),
                                    ROOT.TVector3(0, 1, 0),
                                )
                            ),
                            int(hit_detid[i_z_sorted]),
                        )

                theTrack.insertPoint(ROOT.genfit.TrackPoint(measurement, theTrack))

                # measurement.setMaxDistance(kalman_max_dis[i_z_sorted])
                # measurement.setDetId(int(hit_detid[i_z_sorted]))
                # measurement.setHitId(int(hitID))
                # hitID += 1
                # tp.addRawMeasurement(measurement)
                # theTrack.insertPoint(tp)
                # print(f"Added point: {tp}") # DEBUG

            # if not theTrack.checkConsistency():
            #     #print("Entered first if") # DEBUG
            #     theTrack.Delete()
            #     raise RuntimeError("Kalman fitter track consistency check failed.")

            try:
                theTrack.checkConsistency()
            except Exception as e:
                theTrack.Delete()
                raise RuntimeError("Kalman fitter track consistency check failed.") from e

            # do the fit
            self.kalman_fitter.processTrack(theTrack) # processTrackWithRep(theTrack,rep,True)

            fitStatus = theTrack.getFitStatus()
            if not fitStatus.isFitConverged() and 0>1:
                theTrack.Delete()
                raise RuntimeError("Kalman fit did not converge.")

            # Now save the track if fit converged!
            theTrack.SetUniqueID(self.track_type)
            if fitStatus.isFitConverged():
               print(f"Fit converged! \n")
               if self.genfitTrack: self.kalman_tracks.push_back(theTrack)
               else :
                  # Load items into snd track class object
                  this_track = ROOT.sndRecoTrack(theTrack)
                  pointTimes = ROOT.std.vector(ROOT.std.vector('float'))()
                  for n, i_z_sorted in enumerate(hit_z.argsort()):
                      t_per_hit = []
                      for ch in range(len(hit_time)):
                          if hit_time[ch][i_z_sorted] != None:
                             t_per_hit.append(hit_time[ch][i_z_sorted])
                      pointTimes.push_back(t_per_hit)
                  this_track.setRawMeasTimes(pointTimes)
                  this_track.setTrackType(self.track_type)
                  # Save the track in sndRecoTrack format
                  self.kalman_tracks[i_muon] = this_track
                  # Delete the Kalman track object
                  theTrack.Delete()

            # Remove track hits and try to find an additional track
            # Find array index to be removed
            index_to_remove_ZX = np.where(np.in1d(hit_collection["detectorID"], hit_collection["detectorID"][vertical_condition][track_hits_ZX]))[0]
            index_to_remove_ZY = np.where(np.in1d(hit_collection["detectorID"], hit_collection["detectorID"][horizontal_condition][track_hits_ZY]))[0]

            index_to_remove = np.concatenate([index_to_remove_ZX, index_to_remove_ZY])
            
            # Remove dictionary entries 
            for key in hit_collection.keys() :
                if len(hit_collection[key].shape) == 1 :
                    hit_collection[key] = np.delete(hit_collection[key], index_to_remove)
                elif len(hit_collection[key].shape) == 2 :
                    hit_collection[key] = np.delete(hit_collection[key], index_to_remove, axis = 1)
                else :
                    raise Exception("Wrong number of dimensions found when deleting hits in iterative muon identification algorithm.")

    def FinishTask(self) :
        # Save the ntuple to a file
        self.f_ntuple = ROOT.TFile.Open("f_HT_prediction.root", "recreate")
        self.ntuple.Write()
        self.f_ntuple.Close()
        print("Processed" ,self.events_run)
        if not self.genfitTrack : self.kalman_tracks.Delete()
        else : pass

    # this is a copy on SndlhcTracking function with small adjustments in event object names to make it work here
    # FIXME Should find a way to use this function straight from the SndlhcTracking!
    def scifiCluster(self):
       clusters = []
       hitDict = {}
       for k in range(self.ScifiHits.GetEntries()):
            d = self.ScifiHits[k]
            if not d.isValid(): continue
            hitDict[d.GetDetectorID()] = k
       hitList = list(hitDict.keys())
       if len(hitList)>0:
              hitList.sort()
              tmp = [ hitList[0] ]
              cprev = hitList[0]
              ncl = 0
              last = len(hitList)-1
              hitvector = ROOT.std.vector("sndScifiHit*")()
              for i in range(len(hitList)):
                   if i==0 and len(hitList)>1: continue
                   c=hitList[i]
                   neighbour = False
                   if (c-cprev)==1:    # does not account for neighbours across sipms
                        neighbour = True
                        tmp.append(c)
                   if not neighbour  or c==hitList[last]:
                        first = tmp[0]
                        N = len(tmp)
                        hitvector.clear()
                        for aHit in tmp: hitvector.push_back( self.ScifiHits[hitDict[aHit]])
                        aCluster = ROOT.sndCluster(first,N,hitvector,self.scifiDet,False)
                        clusters.append(aCluster)
                        if c!=hitList[last]:
                             ncl+=1
                             tmp = [c]
                        elif not neighbour :   # save last channel
                            hitvector.clear()
                            hitvector.push_back(self.ScifiHits[hitDict[c]])
                            aCluster = ROOT.sndCluster(c,1,hitvector,self.scifiDet,False)
                            clusters.append(aCluster)
                   cprev = c
       self.clusScifi.Delete()
       for c in clusters:  
            self.clusScifi.Add(c)
