import pandas as pd
import cv2
from pathlib import Path

def load_humanplotted_2d_points(imgptpair_filepath, lengths_filepath, left_fname):
    # Read EventMeasure exports A
    points = pd.read_csv(imgptpair_filepath, sep="\t")
    lengths = pd.read_csv(lengths_filepath, sep="\t")

    # Keep only the frame lookup information
    frame_info = lengths[
        [
            "OpCode","FilenameLeft",
            "ImagePtPair",
            "FrameLeft","FrameRight","Comment"
        ]
    ]

    # Merge using both OpCode and ImagePtPair
    df = points.merge(
        frame_info,
        on=["OpCode", "ImagePtPair"],
        how="left",
        validate="many_to_one"
    )

    # Keep only the stripped third comma-separated comment value (Z).
    df["Comment"] = df["Comment"].astype("string").str.split(",").str[2].str.strip()

    # Keep the smaller left/right frame for each row.
    df["Frame"] = df[["FrameLeft", "FrameRight"]].min(axis=1)
    df = df[df["FilenameLeft"] == left_fname]
    print(df)

    # Keep/reorder desired columns
    df = df[[
            "OpCode",
            "ImagePtPair",
            "Index",
            "Lx",
            "Ly",
            "Rx",
            "Ry",
            "Frame",
            "Comment"
            ]]

    return df

def read_synced_frames(left_path: str, right_path: str, frame_offset: int = 0):
    """
    Yields (frame_index, left_frame, right_frame) for every synced frame pair.

    frame_offset: manually-determined offset between the two videos
          - frame_offset > 0 -> the RIGHT video starts ahead of the left;
            that many frames are discarded from the start of the RIGHT video.
          - frame_offset < 0 -> the LEFT video starts ahead; abs(frame_offset)
            frames are discarded from the start of the LEFT video.
          - frame_offset == 0 (default) -> no slicing, original behavior.

    After this, frame_index 0 in the yielded pairs is the first instant both
    videos have in common.
    """
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    print(left_path)
    if not cap_l.isOpened():
        raise FileNotFoundError(f"Could not open LEFT video:\n{left_path}")

    if not cap_r.isOpened():
        raise FileNotFoundError(f"Could not open RIGHT video:\n{right_path}")

    # discard leading frames from whichever video starts earlier
    skip_l = max(0, -frame_offset)
    skip_r = max(0, frame_offset)
    for _ in range(skip_l):
        cap_l.read()
    for _ in range(skip_r):
        cap_r.read()

    idx = 0
    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        yield idx, frame_l, frame_r
        idx += 1
    cap_l.release()
    cap_r.release()

# ---------------------------------------------------------------------------
# Write the EventMeasure import file (ONE LENGTH line per crab)
# ---------------------------------------------------------------------------
def write_length_import(crabs: list, out_path: str, opcode: str, picture_dir: str,
                         left_mov_name: str, right_mov_name: str,
                         left_cam: str, right_cam: str, delimiter: str = "\t"):
    """
    Writes one EventMeasure "Import text (to EMObs)" file. ONE LENGTH line
    per crab, carrying all four points (left top, left bottom, right top,
    right bottom) in that single row. EventMeasure uses the .Cam
    calibration files to compute the real calibrated measurement between
    the two points when you import this file.
    """
    def line(*fields):
        return delimiter.join(str(f) for f in fields)

    out_lines = ["! Auto-generated EventMeasure LENGTH import - basic skeleton"]
    out_lines.append(line("PIC_DIRECTORY", opcode, picture_dir))
    out_lines.append(line("MOVIE_SEQ_L", opcode, 0, 0.0, left_mov_name))
    out_lines.append(line("MOVIE_SEQ_R", opcode, right_mov_name))
    out_lines.append(line("CAMERA_L", opcode, left_cam))
    out_lines.append(line("CAMERA_R", opcode, right_cam))

    for crab in crabs:
        lt, lb = crab.left_top, crab.left_bottom
        rt, rb = crab.right_top, crab.right_bottom
        out_lines.append(line(
            "LENGTH", opcode,
            left_mov_name, crab.frame, lt[0], lt[1], lb[0], lb[1],
            right_mov_name, crab.frame, rt[0], rt[1], rb[0], rb[1],
            "", "", "",             # Family, Genus, Species
            crab.crab_id,            # Code
            "1",                     # Number
            "", "",                  # Stage, Activity
            "", "", "",              # Attribute8, 9, 10 (unused for now)
        ))

    Path(out_path).write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {len(crabs)} crab(s) = {len(crabs)} LENGTH rows to {out_path}")
