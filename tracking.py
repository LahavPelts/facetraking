"""
Face tracking - computer-only version.

Original project sent the face's (x, y) position over serial to an
Arduino, which drove pan/tilt servos to physically follow the face.

This version removes the Arduino/serial dependency entirely. Instead of
sending motor commands, it computes how far the face is from the center
of the frame (the exact numbers that used to go to the servos) and
displays that as a crosshair + readout on screen.
"""

from cvzone.FaceDetectionModule import FaceDetector
import cv2


def open_camera():
    """
    Try common camera indices so this works whether you have a laptop
    webcam (usually index 0) or an external one (often index 1, which
    the original code hard-coded).
    """
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index {index}")
            return cap
        cap.release()
    raise RuntimeError("No camera found. Try a different index or check connections.")


def main():
    cap = open_camera()
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        img, bboxs = detector.findFaces(img)
        h, w, _ = img.shape
        frame_center = (w // 2, h // 2)

        # Reference crosshair at the frame's true center
        cv2.drawMarker(img, frame_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        if bboxs:
            bbox = bboxs[0]["bbox"]
            center = bboxs[0]["center"]
            x, y, bw, bh = bbox

            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # Offset from center: this is the same data that used to be
            # sent to the Arduino as 'X{x}Y{y}'. Positive dx = face is
            # right of center, positive dy = face is below center.
            dx = center[0] - frame_center[0]
            dy = center[1] - frame_center[1]

            cv2.putText(img, f"dx={dx} dy={dy}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

            # If/when you want to *do* something with dx, dy (move a
            # digital crop window, drive some other output, etc.),
            # this is the spot to add it.

        cv2.imshow("Face Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()