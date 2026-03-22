"""
Real-Time Facial Emotion Detector

This program uses a webcam to detect facial emotions
using DeepFace
 
DeepFace works by looking at a face and comparing it against patterns
it learned from thousands of labeled photos of people expressing different
emotions. It then makes its best guess at what its feeling.
 
    DeepFace (GitHub):  https://github.com/serengil/deepface
    OpenCV (tutorials): https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
 
Required packages:
    pip install deepface opencv-python
 
Press 'q' while the webcam window is open to quit.
"""
import cv2, pyttsx3, threading, queue, time
from deepface import DeepFace


#  Facial Emotion Detection

class EmotionDetector:
    """
    Handles all the emotion detection logic.

    DeepFace can detect 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.
    It analyzes a frame from the webcam and tells us which one it thinks we're showing.

    Since running DeepFace on every single frame would make the program really slow
    we only run it every Nth frame and reuse the last result in between.
    """

    def __init__(self):
        self.analysis_interval = 10   # only analyze every Nth frame
        self.frame_count = 0          # we use this to know when we've hit the Nth frame
        self.last_emotion = "unknown"

    def _select_best_face(self, results, frame_width, frame_height):
        """
        DeepFace can detect multiple faces at once, but we only want one emotion,
        so we pick which face to focus on based on recent one.

        Rather than just grabbing the first one in the list (which has no real
        order), we find whichever face is closest to the center of the screen.
        That way you just point it at the person's voice!

        DeepFace tells us where each face is via a region of the dictionary
            x, y    is the top-left corner of the face box
            w, h    is the width and height of the face box
        We use these to calculate the center of each face, then pick the one
        closest to the center of the frame using the Pythagorean theorem (a² + b² = c²).
        """
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        def distance_to_frame_center(result):
            area = result["region"]
            # find the middle of this particular face
            face_center_x = area["x"] + area["w"] / 2
            face_center_y = area["y"] + area["h"] / 2
            # straight-line distance from face center to frame center
            return ((face_center_x - frame_center_x) ** 2 +
                    (face_center_y - frame_center_y) ** 2) ** 0.5
            # Euclidian distance (thank you pythagoras)
        return min(results, key=distance_to_frame_center)

    def detect(self, frame):
        """
        Takes a single webcam frame and returns best guess at the emotion shown.

        If it's not time to run a full analysis yet, just hand back whatever
        we detected last time.

        Parameters -
        frame: numpy.ndarray
            A raw image from the webcam (in BGR color format, which is how OpenCV works anyway).

        Returns -
        str: something like 'happy' or 'neutral', or 'unknown' if we haven't figured anything out yet.
        """
        self.frame_count += 1

        # not our turn yet — return the cached result and move on
        if self.frame_count % self.analysis_interval != 0:
            return self.last_emotion

        try:
            # Deepface is handed a frame and analyzes it.
            # enforce_detection=False means it won't throw an error if no face is found,
            # it'll just do its best (or return nothing useful).
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],      # skip age/gender/race, only focus on emotion
                enforce_detection=False,  # don't crash if no face is visible
                silent=True               # stop DeepFace from spamming console CONSTANTLY
            )

            # pick the most centered face rather than a random one
            frame_height, frame_width = frame.shape[:2] # only get the height and width, ignore color channels
            best_result = self._select_best_face(results, frame_width, frame_height)

            # DeepFace gives scores for all 7 emotions. Dominant_emotion is just
            # whichever one scored the highest
            dominant_emotion = best_result["dominant_emotion"]
            self.last_emotion = dominant_emotion

        except Exception as error:
            # something went wrong, so keep the last result
            print(f"[EmotionDetector] Warning: {error}")

        return self.last_emotion


prev_emotion = ""
speech_queue = queue.Queue()

def tts_worker():
    """Thread function to handle text-to-speech."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    while True:
        text = speech_queue.get()
        if text is None:                  # Kill the worker if it's done
            speech_queue.task_done()
            break
        try:
            engine.say(text)
            engine.startLoop(False)
            while engine.isBusy():        # Keep iterating
                engine.iterate()
                time.sleep(0.1)
            engine.endLoop()
        except Exception as e:            # Except error
            print(f"TTS error: {e}")
        speech_queue.task_done()

    engine.stop()


def read_emotion(emotion):
    """Queue an emotion when the emotion changes"""
    global prev_emotion

    if prev_emotion != emotion:
        # Discard any pending items so stale emotions don't pile up
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break
        speech_queue.put(f"Emotion: {emotion}")

    prev_emotion = emotion


#  Webcam Loop
def run_detector():
    """
    The main loop. Opens the webcam and keeps grabbing frames until you quit.
    Each frame gets passed to the EmotionDetector, and the result is printed
    to the console. Easy Peasy Lemon Squeezy.
    """
    camera = cv2.VideoCapture(0)  # 0 means default webcam

    if not camera.isOpened():
        print("Error: Could not open webcam. Is it connected?")
        return

    emotion_detector = EmotionDetector()

    # ======= creates a cool effect separating the user instructions from the rest of the console output
    print("=" * 50)
    print("  Real Time Emotion Detector")
    print("  Press 'q' in the video window to quit.")
    print("=" * 50)

    while True:
        success, frame = camera.read()  # next frame from the webcam

        if not success:
            print("Warning: Failed to read a frame from the webcam.")
            break

        detected_emotion = emotion_detector.detect(frame)

        # only print when we actually have a real result
        if detected_emotion != "unknown":
            print(f"Emotion: {detected_emotion}")
            read_emotion(detected_emotion)

        cv2.imshow("Emotion Detector  (press q to quit)", frame)

        # waitKey(1) waits 1ms for a keypress. CV2 needs this otherwise it freezes
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord("q"):
            print("Exiting. Goodbye!")
            break

    # release the webcam and close the window
    camera.release()
    cv2.destroyAllWindows()


# ======== Start =========

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=False)
tts_thread.start()

run_detector()

# Send the shutdown signal and wait for the thread to finish cleanly
speech_queue.put(None)
speech_queue.join()   # wait for all queued items (including None) to be processed
tts_thread.join()     # then wait for thread to exit