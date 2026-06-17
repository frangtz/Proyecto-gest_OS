import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

COOLDOWN = 1.5
ultimo_gesto = 0


def distancia(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def dedos_levantados(lm):
    dedos = []
    palma = lm[9]
    # Pulgar
    dedos.append(distancia(lm[4], palma) > distancia(lm[3], palma))
    # Resto
    for tip, base in [(8,6), (12,10), (16,14), (20,18)]:
        dedos.append(lm[tip].y < lm[base].y)
    return dedos

def pellizco(lm):
    dist = distancia(lm[4], lm[8])
    return dist < 0.05

def detectar_gesto(lm):
    dedos = dedos_levantados(lm)
    pulgar, indice, medio, anular, menique = dedos
    levantados = sum(dedos[1:])  # no cuenta el pulgar para mano abierta

    # Pulgar abajo — chequeo primero antes que mano abierta
    pulgar_apunta_abajo = lm[4].y > lm[3].y > lm[2].y > lm[1].y
    pulgar_extendido = distancia(lm[4], lm[9]) > 0.12
    if pulgar_apunta_abajo and pulgar_extendido:
        return "PULGAR ABAJO", "left"

    # Mano abierta — 4 dedos (sin contar pulgar) levantados
    if levantados >= 4:
        return "MANO ABIERTA", "space"

    # Pulgar arriba — pulgar apunta hacia arriba Y resto claramente cerrado
    resto_cerrado = all(lm[tip].y > lm[base].y for tip, base in [(8,6),(12,10),(16,14),(20,18)])
    if pulgar and resto_cerrado and lm[4].y < lm[2].y:
        return "PULGAR ARRIBA", "right"
    
    # Dos dedos en V — Zoom in
    if indice and medio and not any([pulgar, anular, menique]):
        return "DOS DEDOS", "zoom_in"

    # Tres dedos — Zoom out
    if indice and medio and anular and not any([pulgar, menique]):
        return "TRES DEDOS", "zoom_out"

    # Índice solo
    if indice and not any([pulgar, medio, anular, menique]):
        return "INDICE", "down"
    
    # Pellizco — click izquierdo
    if pellizco(lm):
        return "PELLIZCO", "click"

    # Índice + pulgar levantados — flecha arriba
    if indice and pulgar and not any([medio, anular, menique]):
        if lm[4].y < lm[2].y:  # pulgar apunta arriba
            return "INDICE + PULGAR", "up"

    return None, None

def dibujar_panel(frame, gesto, tecla):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    acciones = dict([
        ("right", "Siguiente Diapo"),         
        ("left",  "Diapo anterior"),
        ("space", "Pausa / Reproducir"),
        ("down",  "Abajo"),
        ("click", "Click izquierdo"),
        ("zoom_in",  "Zoom In"),
        ("zoom_out", "Zoom Out"),
        ("up", "Flecha arriba"),
    ])

    if gesto:
        cv2.putText(frame, f"Gesto: {gesto}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Accion: {acciones.get(tecla, '')}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Esperando gesto...", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    h, w, _ = frame.shape
    cv2.putText(frame, "Mano abierta=Pausa | Pulgar arriba=Sig | Pulgar abajo=Ant | Indice=Flecha-",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesto_actual = None
        tecla_actual = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                gesto_actual, tecla_actual = detectar_gesto(lm)

                ahora = time.time()
                if gesto_actual and (ahora - ultimo_gesto) > COOLDOWN:
                    if tecla_actual == "click":
                        pyautogui.click()
                    elif tecla_actual == "zoom_in":
                        sw, sh = pyautogui.size()
                        pyautogui.keyDown("ctrl")
                        pyautogui.scroll(70, x=sw//2, y=sh//2)
                        pyautogui.keyUp("ctrl")
                    elif tecla_actual == "zoom_out":
                        sw, sh = pyautogui.size()
                        pyautogui.keyDown("ctrl")
                        pyautogui.scroll(-70, x=sw//2, y=sh//2)
                        pyautogui.keyUp("ctrl")
                    else:
                        pyautogui.press(tecla_actual)
                    ultimo_gesto = ahora

        dibujar_panel(frame, gesto_actual, tecla_actual)

        cv2.imshow("Control por Gestos", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()