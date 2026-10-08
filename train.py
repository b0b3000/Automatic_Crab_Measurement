"""
    STUB. Replace this with the real model. For every crab that appears
    anywhere in the video, return one CrabLandmarks: its single best frame,
    plus matched top/bottom pixel coordinates in both views.

    Suggested build order:
      1. Detect + track crabs across left frames (e.g. YOLOv8-seg +
         ByteTrack) -> one track per physical crab.
      2. Per track, score every frame (sharpness, size, detection
         confidence) and keep the single best one.
      3. Segment the crab in that frame -> get a mask.
      4. From the mask, find "top" and "bottom" pixel coordinates
         (however you define body top/bottom - e.g. the two extreme
         points along the body's long axis).
      5. Find the SAME two physical points in the corresponding right
         frame (the stereo-correspondence problem - naive epipolar
         row-scan vs. independently segmenting + matching the right
         frame; worth validating carefully against real footage either
         way, this is the easiest place for silent, hard-to-notice error).
"""

def load_calibration(LEFT_CAM, RIGHT_CAM):
    return


def build_dataset(LEFT_MOV, RIGHT_MOV, STEREO_OFFSET, points_df, calibration):
    return


def split(dataset):
    return


def build_detector_and_tracker():
    return


def build_endpoint_predictor(measurement_types=dataset.measurement_types):
    return


def fit(model, train_set, test_set, loss="unordered_stereo_endpoints"):
    return

#part of this function should visualise the targets vs preds
def evaluate(model, test_set):
    return 


def save_pipeline(detector, tracker, model, calibration, name):
    return

