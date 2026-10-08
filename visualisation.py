import cv2
import pandas as pd
import utils

def visualise_targets(left, right, offset, targets_df, skip, preds_df=None):

    # Target lookup
    targets_by_frame = {
        frame: group.sort_values("Index")
        for frame, group in targets_df.groupby("Frame")
    }

    target_frames = set(targets_by_frame.keys())

    # Optional prediction lookup
    preds_by_frame = {}
    pred_frames = set()

    if preds_df is not None:
        preds_by_frame = {
            frame: group.sort_values("Index")
            for frame, group in preds_df.groupby("Frame")
        }

        pred_frames = set(preds_by_frame.keys())

    for idx, frame_l, frame_r in utils.read_synced_frames(
        left,
        right,
        offset
    ):

        target_comment = None
        pred_comment = None
        target_set = None
        pred_set = None

        # Human target points
        if idx in target_frames:
            pts = targets_by_frame[idx]
            p0, p1 = pts.iloc[0], pts.iloc[1]

            target_comment = p0.get("Comment")

            target_set = [
                (p0.Lx, p0.Ly),
                (p1.Lx, p1.Ly),
                (p0.Rx, p0.Ry),
                (p1.Rx, p1.Ry)
            ]

        # Predicted points
        if idx in pred_frames:
            pts = preds_by_frame[idx]
            p0, p1 = pts.iloc[0], pts.iloc[1]

            pred_comment = p0.get("Comment")

            pred_set = [
                (p0.Lx, p0.Ly),
                (p1.Lx, p1.Ly),
                (p0.Rx, p0.Ry),
                (p1.Rx, p1.Ry)
            ]

        # Show frame if:
        # - it has targets
        # - it has predictions
        # - or it is every skip-th frame
        if target_set is not None or pred_set is not None:

            key = show_stereo_frame(
                frame_l,
                frame_r,
                idx,
                0.25,
                target_points=target_set,
                pred_points=pred_set,
                target_comment=target_comment,
                pred_comment=pred_comment
            )

        elif idx % skip == 0:

            key = show_stereo_frame(
                frame_l,
                frame_r,
                idx,
                scale=0.25
            )

        else:
            continue

        if key == ord("q"):
            break

    close_visualisation()


def show_stereo_frame(
    frame_l,
    frame_r,
    idx,
    scale=0.25,
    target_points=None,
    pred_points=None,
    target_comment=None,
    pred_comment=None,
):
    """
    Display left and right stereo frames side-by-side.

    target_points and pred_points should each be:
        (
            left_p0,
            left_p1,
            right_p0,
            right_p1
        )

    where each point is an (x, y) tuple in ORIGINAL image coordinates.

    Optional target_comment/pred_comment provide one comment per point set,
    displayed alongside its legend label.

    Colours:
        target_points -> blue
        pred_points   -> red
    """

    frame_l_display = cv2.resize(
        frame_l,
        None,
        fx=scale,
        fy=scale
    )

    frame_r_display = cv2.resize(
        frame_r,
        None,
        fx=scale,
        fy=scale
    )

    # OpenCV uses BGR
    target_colour = (255, 0, 0)   # blue
    pred_colour   = (0, 0, 255)   # red

    point_sets = [
        (target_points, target_colour, target_comment),
        (pred_points, pred_colour, pred_comment),
    ]

    for point_set, colour, comments in point_sets:

        if point_set is None:
            continue

        left_p0, left_p1, right_p0, right_p1 = point_set

        # Scale LEFT points
        lp0 = (
            int(left_p0[0] * scale),
            int(left_p0[1] * scale)
        )

        lp1 = (
            int(left_p1[0] * scale),
            int(left_p1[1] * scale)
        )

        # Scale RIGHT points
        rp0 = (
            int(right_p0[0] * scale),
            int(right_p0[1] * scale)
        )

        rp1 = (
            int(right_p1[0] * scale),
            int(right_p1[1] * scale)
        )

        # LEFT
        cv2.circle(frame_l_display, lp0, 2, colour, -1)
        cv2.circle(frame_l_display, lp1, 2, colour, -1)
        cv2.line(frame_l_display, lp0, lp1, colour, 1)

        # RIGHT
        cv2.circle(frame_r_display, rp0, 2, colour, -1)
        cv2.circle(frame_r_display, rp1, 2, colour, -1)
        cv2.line(frame_r_display, rp0, rp1, colour, 1)

    cv2.putText(
        frame_l_display,
        f"LEFT - frame {idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame_r_display,
        f"RIGHT - frame {idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Draw legend if at least one point set is present
    if target_points is not None or pred_points is not None:

        legend_items = []
        if target_points is not None:
            legend_items.append(("Target", target_colour, target_comment))
        if pred_points is not None:
            legend_items.append(("Prediction", pred_colour, pred_comment))

        y = 55
        for label, colour, comment in legend_items:
            if comment is not None and not pd.isna(comment):
                comment = str(comment).strip()
                if comment:
                    label = f"{label}: {comment}"
            # small coloured line
            cv2.line(frame_l_display, (10, y), (25, y), colour, 2)
            # small coloured dot
            cv2.circle(frame_l_display, (17, y), 3, colour, -1)
            # label text
            cv2.putText(
                frame_l_display,
                label,
                (30, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1
            )
            y += 20

    combined = cv2.hconcat([
        frame_l_display,
        frame_r_display
    ])

    cv2.imshow("Stereo Sync Test", combined)

    key = cv2.waitKey(0) & 0xFF

    return key


def close_visualisation():
    cv2.destroyAllWindows()


def print_target(points_df):

    for (opcode, pair_id), group in points_df.head(10).groupby(
        ["OpCode", "ImagePtPair"]
    ):
        print(f"\n{opcode} | ImagePtPair {pair_id}")

        for _, row in group.iterrows():
            print(
                f"  Point {int(row['Index'])}: "
                f"L=({row['Lx']:.2f}, {row['Ly']:.2f})  "
                f"R=({row['Rx']:.2f}, {row['Ry']:.2f})  "
                f"Frame={int(row['Frame'])}"
            )