from ultralytics import YOLO
import cv2
from cv2.typing import MatLike
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from typing import Any, TextIO
import jsonpickle  # type: ignore

FORGET_AFTER_UNSEEN_FRAMES = 120
SAVE_EVERY_N_FRAMES = 120


def request_n_lines(
    image: MatLike, n: int = 1
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Displays an image and requires the user to draw 'n' lines.
    - 'e': Undo last line
    - Enter: Submit (only works if 'n' lines are drawn)
    - Esc/Q: Cancel
    """
    state: dict[str, Any] = {
        "lines": [],  # Stores [(p1, p2), (p1, p2), ...]
        "current_start": None,
        "current_end": None,
        "drawing": False,
        "base_image": image.copy(),
        "window_name": "Image Window",
        "target_n": n,
    }

    def update_title():
        remaining = state["target_n"] - len(state["lines"])
        if remaining > 0:
            title = f"Draw {remaining} more | 'e': Undo"
        else:
            title = "Done! Press ENTER to submit | 'e': Undo"
        cv2.setWindowTitle(state["window_name"], title)

    def redraw():
        # Refresh the frame from the original image
        img_copy = state["base_image"].copy()

        # Draw all saved lines
        for pt1, pt2 in state["lines"]:
            cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)

        # Draw the active "rubber band" line
        if state["drawing"] and state["current_start"] and state["current_end"]:
            cv2.line(
                img_copy, state["current_start"], state["current_end"], (0, 255, 255), 2
            )

        cv2.imshow(state["window_name"], img_copy)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["drawing"] = True
            param["current_start"] = (x, y)
            param["current_end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if param["drawing"]:
                param["current_end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            param["drawing"] = False
            # Prevent saving a zero-length line (accidental click)
            if param["current_start"] != (x, y):
                param["lines"].append((param["current_start"], (x, y)))
            if len(param["lines"]) >= param["target_n"]:
                cv2.setMouseCallback(state["window_name"], lambda *args: None)  # type: ignore
        update_title()
        redraw()

    # Window Initialization
    cv2.namedWindow(state["window_name"])
    update_title()
    redraw()
    cv2.setMouseCallback(state["window_name"], mouse_callback, param=state)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 1. Enter key (ASCII 13 or 10) - Submit
        if key in [13, 10]:
            if len(state["lines"]) >= state["target_n"]:
                break
            else:
                print(
                    f"Required: {state['target_n']} lines. Current: {len(state['lines'])}"
                )

        # 2. 'e' key - Undo
        elif key == ord("e"):
            if state["lines"]:
                state["lines"].pop()
                update_title()
                redraw()
                if len(state["lines"]) < state["target_n"]:
                    cv2.setMouseCallback(
                        state["window_name"], mouse_callback, param=state
                    )

        # 3. Esc (27) or 'q' - Quit/Cancel
        elif key in [27, ord("q")]:
            state["lines"] = []  # Clear results on cancel
            break

    # Cleanup: Uninstall callback and close window
    cv2.setMouseCallback(state["window_name"], lambda *args: None)  # type: ignore
    cv2.destroyWindow(state["window_name"])

    # Just in case we have more lines than we need.
    return state["lines"][: state["target_n"]]  # type: ignore


def get_valid_video_path() -> str:
    """
    Prompts the user for a video file path and validates it.
    Returns the valid file path once confirmed.
    """
    while True:
        video_path = input("Enter path to video file: ").strip()

        if os.path.isfile(video_path):
            print(f"✓ Valid file found: {video_path}")
            return video_path
        else:
            print(f"✗ File not found: {video_path}")
            print("Please try again.\n")


def get_first_frame(video_path: str) -> MatLike | None:
    """
    Opens a video file and returns the first frame.

    Args:
        video_path (str): Path to the video file

    Returns:
        np.ndarray: The first frame of the video, or None if unable to read
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"✗ Error: Could not open video file: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("✗ Error: Could not read frame from video")
        return None

    print("✓ Successfully read first frame from video")
    return frame


@dataclass
class Point:
    x: float
    y: float


def segments_intersect(line1: tuple[Point, Point], line2: tuple[Point, Point]) -> bool:
    """
    Returns True if line segment 1 intersects with line segment 2.
    """

    def _on_segment(p: Point, q: Point, r: Point) -> bool:
        """
        Given three collinear points p, q, and r, checks if point q
        lies on the line segment bounded by p and r.
        """
        return (min(p.x, r.x) <= q.x <= max(p.x, r.x)) and (
            min(p.y, r.y) <= q.y <= max(p.y, r.y)
        )

    def _orientation(p: Point, q: Point, r: Point) -> int:
        """
        Finds the orientation of an ordered triplet (p, q, r).
        Returns:
        0: p, q, and r are collinear
        1: Clockwise
        2: Counterclockwise
        """
        # Cross product of segments pq and qr
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

        if val > 0:
            return 1  # Clockwise
        elif val < 0:
            return 2  # Counterclockwise
        else:
            return 0  # Collinear

    p1, q1 = line1
    p2, q2 = line2

    # Find the four orientations needed for the general and special cases
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # 1. General Case
    # The segments intersect if p2 and q2 have different orientations relative to p1q1,
    # AND p1 and q1 have different orientations relative to p2q2.
    if o1 != o2 and o3 != o4:
        return True

    # 2. Special Cases (Collinear intersections)
    # p1, q1 and p2 are collinear, and p2 lies on segment p1q1
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear, and q2 lies on segment p1q1
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear, and p1 lies on segment p2q2
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear, and q1 lies on segment p2q2
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    # If none of the above cases apply, the segments do not intersect
    return False


class SingleLineCondition:
    def __init__(self, line: tuple[tuple[int, int], tuple[int, int]]):
        self._line = Point(float(line[0][0]), float(line[0][1])), Point(
            float(line[1][0]), float(line[1][1])
        )

    def test(self, id: int, frame: int, past_point: Point, curr_point: Point) -> bool:
        return segments_intersect(self._line, (past_point, curr_point))

    def forget(self, id: int):
        return


class DoubleLineCondition:
    def __init__(
        self,
        line_1: tuple[tuple[int, int], tuple[int, int]],
        line_2: tuple[tuple[int, int], tuple[int, int]],
    ):
        self._line_1 = Point(float(line_1[0][0]), float(line_1[0][1])), Point(
            float(line_1[1][0]), float(line_1[1][1])
        )
        self._line_2 = Point(float(line_2[0][0]), float(line_2[0][1])), Point(
            float(line_2[1][0]), float(line_2[1][1])
        )
        self._line_1_passed: dict[int, int] = {}

    def test(self, id: int, frame: int, past_point: Point, curr_point: Point) -> bool:
        if segments_intersect(self._line_1, (past_point, curr_point)):
            self._line_1_passed[id] = frame
        if segments_intersect(self._line_2, (past_point, curr_point)):
            if id in self._line_1_passed:
                self._line_1_passed.pop(id)
                return True
        return False

    def forget(self, id: int):
        self._line_1_passed.pop(id, None)


def add_single_line_condition(frame: MatLike) -> SingleLineCondition | None:
    print("Draw a single line on the image.")
    print("All objects that pass the line will be counted.")
    lines = request_n_lines(frame, n=1)
    if len(lines) != 1:
        return None
    return SingleLineCondition(lines[0])


def add_double_line_condition(frame: MatLike) -> DoubleLineCondition | None:
    print("Draw two lines on the image.")
    print("Objects that pass Line 1 first and then pass Line 2 will be counted.")
    lines = request_n_lines(frame, n=2)
    if len(lines) != 2:
        return None
    return DoubleLineCondition(lines[0], lines[1])


if __name__ == "__main__":
    config: dict[str, Any] = {}
    print("Specify a config file if you have one.")
    option = input(
        "Enter path to config file, or leave it blank for manual configuration: "
    )
    config_path = Path(option)
    try:
        config = jsonpickle.decode(config_path.open().read())  # type: ignore
        print("Succesfully loaded config.")
    except Exception:
        print("Failed to load config file. Proceeding to manual configuration.")

        print("Specify a video file to analyze.")
        video_path = get_valid_video_path()
        config["video_path"] = video_path
        first_frame = get_first_frame(video_path)
        if first_frame is None:
            print("Failed to fetch the first frame of the video.")
            raise Exception

        conditions: dict[str, SingleLineCondition | DoubleLineCondition] = {}

        while True:
            print("Specify counting conditions.")
            print("1. Single line: objects that pass the line will be counted.")
            print(
                "2. Two lines: objects that pass line 1 first and then pass line 2 will be counted."
            )
            print("q. Proceed to counting.")
            option = input('Write "1", "2", or "q", and press Enter: ')
            if option == "1":
                condition = add_single_line_condition(first_frame)
            elif option == "2":
                condition = add_double_line_condition(first_frame)
            elif option == "q":
                break
            else:
                continue
            if condition is not None:
                while True:
                    print("Give this condition a name.")
                    condition_name = input("Enter name: ")
                    if condition_name in conditions:
                        print("Duplicate names are not allowed.")
                        continue
                    conditions[condition_name] = condition
                    break
        config["conditions"] = conditions

        while True:
            print("Specify video stride.")
            print("Stride of 2 means the model will only process once every 2 frames.")
            print("High video stride -> fast but inaccurate.")
            option = input("Write a number, and press Enter: ")
            if not option.isdigit():
                continue
            config["video_stride"] = int(option)
            break

        while True:
            print("Specify a model.")
            print("ultralytics models are trained for general object detection.")
            print("1. ultralytics/yolo26n")
            print("2. ultralytics/yolo26s")
            print("3. ultralytics/yolo26m")
            print("4. ultralytics/yolo26l")
            print("5. ultralytics/yolo26x")
            print(
                "Perception365 models are fine-tuned for traffic detection, but requires access to a gated repo."
            )
            print("6. Perception365/VehicleNet-Y26n")
            print("7. Perception365/VehicleNet-Y26s")
            print("8. Perception365/VehicleNet-Y26m")
            print("9. Perception365/VehicleNet-Y26x")
            option = input('Write "1" - "9", and press Enter: ')
            if option == "1":
                config["model_name"] = "ultralytics/yolo26n"
                break
            elif option == "2":
                config["model_name"] = "ultralytics/yolo26s"
                break
            elif option == "3":
                config["model_name"] = "ultralytics/yolo26m"
                break
            elif option == "4":
                config["model_name"] = "ultralytics/yolo26l"
                break
            elif option == "5":
                config["model_name"] = "ultralytics/yolo26x"
                break
            elif option == "6":
                config["model_name"] = "Perception365/VehicleNet-Y26n"
                break
            elif option == "7":
                config["model_name"] = "Perception365/VehicleNet-Y26s"
                break
            elif option == "8":
                config["model_name"] = "Perception365/VehicleNet-Y26m"
                break
            elif option == "9":
                config["model_name"] = "Perception365/VehicleNet-Y26x"
                break
            else:
                continue

    now = datetime.now()
    unique_id = now.strftime("%Y%m%d-%H%M%S")

    config_path = Path(f"./config/{unique_id}.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open(mode="x") as config_file:
        config_file.write(jsonpickle.encode(config))  # type: ignore

    condition_dataframes: dict[str, pd.DataFrame] = {}
    condition_files: dict[str, TextIO] = {}
    for condition_name in config["conditions"].keys():
        output_path = Path(f"./outputs/{condition_name}-{unique_id}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open(mode="x")
        condition_dataframes[condition_name] = pd.DataFrame(
            columns=["frame", "id", "class"]
        )
        condition_files[condition_name] = output_file  # type: ignore
        condition_dataframes[condition_name].to_csv(
            output_file, mode="a", index=False, header=True
        )

    if config["model_name"] == "ultralytics/yolo26n":
        model = YOLO("yolo26n.pt")
    elif config["model_name"] == "ultralytics/yolo26s":
        model = YOLO("yolo26s.pt")
    elif config["model_name"] == "ultralytics/yolo26m":
        model = YOLO("yolo26m.pt")
    elif config["model_name"] == "ultralytics/yolo26l":
        model = YOLO("yolo26l.pt")
    elif config["model_name"] == "ultralytics/yolo26x":
        model = YOLO("yolo26x.pt")
    elif "Perception365" in config["model_name"]:
        from huggingface_hub import login, hf_hub_download  # type: ignore

        login(skip_if_logged_in=True)

        if config["model_name"] == "Perception365/VehicleNet-Y26n":
            model = YOLO(
                hf_hub_download(
                    repo_id="Perception365/VehicleNet-Y26n",
                    filename="weights/best.pt",
                )
            )
        elif config["model_name"] == "Perception365/VehicleNet-Y26s":
            model = YOLO(
                hf_hub_download(
                    repo_id="Perception365/VehicleNet-Y26s",
                    filename="weights/best.pt",
                )
            )
        elif config["model_name"] == "Perception365/VehicleNet-Y26m":
            model = YOLO(
                hf_hub_download(
                    repo_id="Perception365/VehicleNet-Y26m",
                    filename="weights/best.pt",
                )
            )
        elif config["model_name"] == "Perception365/VehicleNet-Y26x":
            model = YOLO(
                hf_hub_download(
                    repo_id="Perception365/VehicleNet-Y26x",
                    filename="weights/best.pt",
                )
            )
        else:
            raise Exception
    else:
        raise Exception
    if "ultralytics" in config["model_name"]:
        results_gen = model.track(  # type: ignore
            source=config["video_path"],
            show=True,
            conf=0.05,
            imgsz=(1920, 1080),
            # batch=16,
            vid_stride=config["video_stride"],
            classes=[2, 3, 5, 7],
            stream=True,
            tracker="botsort.yaml",
        )
    else:
        results_gen = model.track(  # type: ignore
            source=config["video_path"],
            show=True,
            conf=0.02,
            imgsz=(1920, 1080),
            # batch=16,
            vid_stride=config["video_stride"],
            # classes=[2, 3, 5, 7],
            stream=True,
            tracker="botsort.yaml",
        )

    last_known_positions: dict[int, tuple[int, Point]] = {}
    conditions = config["conditions"]

    for frame, results in enumerate(results_gen):
        if frame % SAVE_EVERY_N_FRAMES == 0:
            for condition_name, condition_dataframe in condition_dataframes.items():
                condition_dataframe.to_csv(
                    condition_files[condition_name],
                    mode="a",
                    index=False,
                    header=False,
                )
                condition_files[condition_name].flush()
                condition_dataframes[condition_name] = pd.DataFrame(
                    columns=["frame", "id", "class"]
                )
        if frame % FORGET_AFTER_UNSEEN_FRAMES == 0:
            new_last_known_positions: dict[int, tuple[int, Point]] = {}
            for k, v in last_known_positions.items():
                if (
                    frame - 1 - v[0] >= FORGET_AFTER_UNSEEN_FRAMES
                ):  # If last seen at frame 20, and we are at the start of frame 25, the object wasn't seen between frame 21 - 24 = 4 frames.
                    for condition in conditions.values():
                        condition.forget(k)
                else:
                    new_last_known_positions[k] = v
            last_known_positions = new_last_known_positions

        for result in results:
            if not result.boxes or not result.boxes.is_track:  # type: ignore
                continue

            box = result.boxes.xywh.cpu()  # type: ignore
            track_id = int(result.boxes.id.int().item())  # type: ignore
            class_label = result.boxes.cls.int().item()  # type: ignore
            x, y, w, h = box[0, :]  # type: ignore
            curr_pos = Point(float(x.item()), float(y.item()))  # type: ignore
            if track_id in last_known_positions.keys():
                _, past_pos = last_known_positions[track_id]
                for condition_name, condition in conditions.items():
                    if condition.test(track_id, frame, past_pos, curr_pos):  # type: ignore
                        new_row = pd.DataFrame(
                            [[frame * config["video_stride"], track_id, class_label]],
                            columns=["frame", "id", "class"],
                        )
                        condition_dataframes[condition_name] = pd.concat(
                            [condition_dataframes[condition_name], new_row],
                            ignore_index=True,
                        )
                        print(
                            f"{frame}: {condition_name} satisfied for object {track_id}."
                        )

            last_known_positions[track_id] = (frame, curr_pos)
