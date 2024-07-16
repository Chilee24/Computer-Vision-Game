import cv2
import mediapipe as mp
import random
from typing import NamedTuple, Optional, List

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
score = 0

x_enemy = random.randint(50, 600)
y_enemy = random.randint(50, 400)


def enemy(image):
    cv2.circle(image, (x_enemy, y_enemy), 25, (0, 200, 0), 5)


def display_score(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 255)
    cv2.putText(image, "Score:", (480, 30), font, 1, color, 4, cv2.LINE_AA)
    cv2.putText(image, str(score), (590, 30), font, 1, color, 4, cv2.LINE_AA)


def check_collision(cx, cy):
    global score, x_enemy, y_enemy
    if abs(cx - x_enemy) < 25 and abs(cy - y_enemy) < 25:
        score += 1
        x_enemy = random.randint(50, 600)
        y_enemy = random.randint(50, 400)


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results: NamedTuple = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            enemy(image)
            display_score(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    for point in mp_hands.HandLandmark:
                        normalized_landmark = hand_landmarks.landmark[point]
                        pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(
                            normalized_landmark.x, normalized_landmark.y, image.shape[1], image.shape[0])

                        if pixel_coordinates_landmark:
                            cx, cy = pixel_coordinates_landmark
                            check_collision(cx, cy)

            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
