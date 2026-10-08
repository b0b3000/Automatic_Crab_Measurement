"""
Basic outline of the pipeline.

INPUT:  working directory
        left/right .mov file names
        left/right .Cam calibration file names
        Left/right stereo frame offset
        output file name
        Opcode TODO What is this?
        Point pair & Lengths (eMOB file exports) file names

MODEL:  for each crab in the video -> find its best/most-visible frame ->
        locate two landmarks (top-of-body, bottom-of-body) as pixel
        coordinates - TWO points in the left frame, TWO matching points in
        the right frame, four numbers-pairs total per crab

OUTPUT: an EventMeasure "Import text (to EMObs)" file, ONE LENGTH line per
        crab, carrying all four pixel points (left top, left bottom, right
        top, right bottom) in that single row. Importing this into
        EventMeasure is what actually produces a real calibrated
        measurement - this script only gets you to the pixel-coordinate
        stage.

Run (once model is trained):
    python pipeline.py
    #TODO maybe we need a main.py or run.py... this is ok for now
"""

import cv2
from dataclasses import dataclass
from pathlib import Path

import utils
import visualisation
import train


# ---------------------------------------------------------------------------
# CONFIG - edit these for your actual files
# ---------------------------------------------------------------------------

# 1) User Inputs ----------------------------------------------------------

DIR = Path(r'C:\Users\snd\Desktop\bobs code\data\24_08_07_Napoleon')
LEFT_MOV_FILE =  "24_08_07NapoleonL1.MOV"
RIGHT_MOV_FILE = "24_08_07NapoleonR1.MOV"
LEFT_CAM_FILE = "Napoleon_L_PreCAL_29052024_HD air.CamCAL"
RIGHT_CAM_FILE = "Napoleon_R_PreCAL_29052024_HD air.CamCAL"
STEREO_OFFSET = -353 #can we find a way to automate this?
OPCODE    = "test1_output" #Not 100% sure what this does
OUT_FILE  = "test1_import.txt"


# 2) Full files -------------------------------------------------
LEFT_MOV  = DIR / LEFT_MOV_FILE
RIGHT_MOV = DIR / RIGHT_MOV_FILE
LEFT_CAM  = DIR / LEFT_CAM_FILE
RIGHT_CAM = DIR / RIGHT_CAM_FILE
PIC_DIR   = DIR #Not 100% sure what this does either


# Training ----------------------------------------------------------
PTPAIRS = DIR / "24_08_07Napoleon_ImagePtPair.txt"
LENGTHS =  DIR / "24_08_07Napoleon_Lengths.txt"

# Visualisation Control ---------------------------------------
FRAME_SKIP = 1000

# ---------------------------------------------------------------------------
# One record per crab: its best frame + all four pixel point pairs
# (two in the left frame, two matching points in the right frame).
# ---------------------------------------------------------------------------
@dataclass
class CrabLandmarks:
    crab_id: str            # unique per crab (used as the Code field)
    frame: int               # best/most-visible frame index (assumes left/right synced)
    left_top: tuple          # (col, row) pixel coords in the LEFT frame
    left_bottom: tuple       # (col, row) pixel coords in the LEFT frame
    right_top: tuple         # (col, row) matched pixel coords in the RIGHT frame
    right_bottom: tuple      # (col, row) matched pixel coords in the RIGHT frame

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    #TRAINING -----------------------------------------------------
    if training:
        points_df = utils.load_humanplotted_2d_points(PTPAIRS, LENGTHS,LEFT_MOV_FILE)
        #visualisation.print_target(points_df)
        #visualisation.visualise_targets(LEFT_MOV, RIGHT_MOV, STEREO_OFFSET, points_df, FRAME_SKIP)

        # Abstract skeleton: helper functions still need implementing.

        calibration = train.load_calibration(LEFT_CAM, RIGHT_CAM)
        dataset = train.build_dataset(LEFT_MOV, RIGHT_MOV, STEREO_OFFSET, points_df, calibration)
        train_set, test_set = train.split(dataset)
        detector, tracker = train.build_detector_and_tracker()
        model = train.build_endpoint_predictor(measurement_types=dataset.measurement_types)
        model = train.fit(model, train_set, test_set, loss="unordered_stereo_endpoints")
        train.evaluate(model, test_set) #part of this function should visualise the targets vs preds
        train.save_pipeline(detector, tracker, model, calibration, "crab_pipeline")


    #ALREADY TESTED -----------------------------------------------------
    else: 
        #TODO Run pipeline with File/Cal inputs, crabs list/df outputs
        crabs = [] # TODO This would come from output of the pipeline
        utils.write_import_file(
                crabs, OUT_FILE,
                opcode=OPCODE, picture_dir=PIC_DIR,
                left_mov_name=Path(LEFT_MOV).name, right_mov_name=Path(RIGHT_MOV).name,
                left_cam=LEFT_CAM, right_cam=RIGHT_CAM,
            )

        #TODO Can we directly pipe the import in in the code? or does it have to be done in EM GUI?
    
    '''
    preds_df = find_crab_landmarks(LEFT_MOV, RIGHT_MOV)  # STUB - raises until wired up'''

#NEXT STEP: