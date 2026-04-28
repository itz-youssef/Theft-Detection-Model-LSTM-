Theft Detection - Behavioral AI
An advanced, real-time computer vision pipeline designed to detect suspicious behavior and potential theft in retail environments. Instead of just identifying objects, this system analyzes human biomechanics, spatial interactions, and temporal patterns to identify the act of stealing.

✨ Overview
This project leverages state-of-the-art deep learning models to track individuals, analyze their skeletal posture, and monitor their interactions with high-value items (like phones, laptops, and bags). By combining rule-based heuristics with a custom Long Short-Term Memory (LSTM) network, the pipeline can distinguish between normal shopping behavior and highly suspicious actions (e.g., rapid grabbing, pocketing items, loitering, and fleeing).

✨ Key Features
Complex Behavioral Analysis: Uses pose estimation keypoints to calculate precise metrics like spine lean, wrist speed, elbow extension, and hand-to-pocket distance.

Temporal LSTM Modeling: Feeds a 20-frame rolling window of extracted skeletal features into a custom PyTorch LSTM (TheftDetectionLSTM) to recognize suspicious movement patterns over time.

High-Value Object Tracking: Monitors specific COCO classes (phones, laptops, bags) and alerts the system if an object disappears immediately after an individual's hands enter its proximity.

Robust Re-Identification (ReID): Integrates a ResNet18-based appearance extractor and color histograms to maintain track identities even when individuals are occluded or leave the camera frame temporarily (LostTrackMemory).

SAHI Integration (Optional): Supports Sliced Aided Hyper Inference (SAHI) to detect tiny objects (like mobile phones on shelves) in high-resolution video streams.

False-Positive Suppression: Features an advanced clustering algorithm that handles "depth occlusion"—preventing innocent bystanders standing behind a suspect from being falsely flagged due to overlapping bounding boxes.

Multi-Model Architecture: Simultaneously runs Ultralytics YOLOv11 (Pose, Segmentation, Object) and ByteTrack for flawless multi-object tracking.

🧠 System Architecture
Detection & Tracking: YOLOv11 extracts bounding boxes, segmentation masks, and 17-point skeletal keypoints. ByteTrack assigns persistent IDs.

Feature Extraction: The SkeletonAnalyzer computes 17 highly specific spatial features (e.g., grab_motion_score, wrist_moving_toward_body, arm_extension).

Appearance Embedding: Batched crops of tracked individuals are passed through a ResNet18 model to generate embeddings for ReID.

Object Interaction: The ObjectTracker calculates the distance between human wrists and tracked high-value items, flagging when a touch occurs or an object vanishes.

Risk Scoring Gate: Extracted features are evaluated by both a heavily calibrated rule-based logic gate and the LSTM classifier.

Alerting: If the combined risk score crosses the threshold, the system flags the individual as a CONFIRMED THIEF and highlights them on the output feed.

🛠️ Tech Stack
Language: Python 3.x

Deep Learning Frameworks: PyTorch, Torchvision

Computer Vision: OpenCV, Ultralytics (YOLOv11)

Tracking & Processing: Supervision (ByteTrack), SAHI

Data Processing: NumPy, SciPy

⚡ Performance & Hardware Note
Because this pipeline runs multiple heavy models sequentially (Pose, Segmentation, Object Detection, ResNet18, and LSTM), it is computationally intensive.

For maximum accuracy: Use the yolo11x (Extra Large) weights and enable SAHI.

For real-time performance (25+ FPS): Switch to yolo11s (Small) or yolo11m (Medium) weights and set USE_SAHI_OBJECT_DETECTION = False in the configuration.
