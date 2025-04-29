import cv2
import numpy as np
import tensorflow as tf
import config # Import constants

def preprocess_for_seatbelt(person_crop):
    """Preprocesses the cropped person image for the seatbelt model."""
    img_resized = cv2.resize(person_crop, config.IMG_SIZE_SEATBELT, interpolation=cv2.INTER_AREA)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    # Normalize if the model expects it (often needed for models trained with ImageNet stats)
    # Example normalization (adjust if your model used different normalization):
    img_array = (img_array / 127.5) - 1.0
    return img_array

def detect_objects_and_seatbelt(frame, person_model, phone_model, seatbelt_model):
    """Detects persons, phones within persons, and classifies seatbelt status."""
    results_person = person_model(frame)
    persons = results_person.pandas().xyxy[0]
    persons = persons[persons['name'] == 'person'] # Filter for persons
    persons = persons[persons['confidence'] > config.CONFIDENCE_THRESHOLD_PERSON]

    # Optional: Apply NMS specifically for persons if needed, though YOLOv5 does this internally.
    # boxes_person = persons[['xmin', 'ymin', 'xmax', 'ymax']].values
    # scores_person = persons['confidence'].values
    # indices_person = tf.image.non_max_suppression(boxes_person, scores_person, max_output_size=50, iou_threshold=config.IOU_THRESHOLD)
    # persons = persons.iloc[indices_person.numpy()]

    detections = []

    for index, person in persons.iterrows():
        px1, py1, px2, py2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])

        # Ensure coordinates are within frame bounds
        h, w, _ = frame.shape
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w - 1, px2), min(h - 1, py2)

        # Crop person region for further processing
        person_crop = frame[py1:py2, px1:px2]

        if person_crop.size == 0:
            continue # Skip if crop is empty

        # --- Seatbelt Classification ---
        seatbelt_status = "Unknown"
        seatbelt_score = 0.0
        try:
            processed_crop = preprocess_for_seatbelt(person_crop)
            prediction = seatbelt_model.predict(processed_crop, verbose=0) # Set verbose=0 to avoid printing Keras progress
            seatbelt_score = np.max(prediction[0])
            predicted_class_index = np.argmax(prediction[0])

            # Use class names from config
            if predicted_class_index < len(config.CLASS_NAMES_SEATBELT):
                 # Only assign status if score is above threshold
                if seatbelt_score >= config.THRESHOLD_SCORE_SEATBELT:
                    seatbelt_status = config.CLASS_NAMES_SEATBELT[predicted_class_index]
                else:
                    seatbelt_status = "Not Worn" # Default to Not Worn if confidence is low
            else:
                print(f"Warning: Predicted class index {predicted_class_index} out of bounds for CLASS_NAMES_SEATBELT.")
                seatbelt_status = "Error"

        except Exception as e:
            print(f"Error during seatbelt prediction: {e}")
            seatbelt_status = "Error"
            seatbelt_score = 0.0

        # --- Phone Detection within Person Crop ---
        phone_detected = False
        phone_box = None
        try:
            results_phone = phone_model(person_crop)
            phones = results_phone.pandas().xyxy[0]
            # Assuming your phone model has a class named 'phone' or similar
            # Adjust 'phone' if your model uses a different class name
            phones = phones[phones['name'] == 'phone']
            phones = phones[phones['confidence'] > config.CONFIDENCE_THRESHOLD_PHONE]

            if not phones.empty:
                phone_detected = True
                # Get the box with the highest confidence (optional)
                best_phone = phones.loc[phones['confidence'].idxmax()]
                # Convert phone box coordinates relative to the original frame
                phx1 = px1 + int(best_phone['xmin'])
                phy1 = py1 + int(best_phone['ymin'])
                phx2 = px1 + int(best_phone['xmax'])
                phy2 = py1 + int(best_phone['ymax'])
                phone_box = [phx1, phy1, phx2, phy2]

        except Exception as e:
            print(f"Error during phone detection: {e}")
            # Continue without phone detection for this person

        detections.append({
            'person_box': [px1, py1, px2, py2],
            'seatbelt_status': seatbelt_status,
            'seatbelt_score': float(seatbelt_score),
            'phone_detected': phone_detected,
            'phone_box': phone_box # Can be None
        })

    return detections
