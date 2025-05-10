import depthai as dai
import config
import os

def load_models():
    """Creates and configures the DepthAI pipeline with blob models."""
    try:
        # Create DepthAI pipeline
        pipeline = dai.Pipeline()
        
        # Set OpenVINO version for optimal performance
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)
        
        # Define sources and outputs
        cam_rgb = pipeline.create(dai.node.XLinkIn)
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_nn = pipeline.create(dai.node.XLinkOut)
        
        cam_rgb.setStreamName("frame")
        xout_rgb.setStreamName("rgb")
        xout_nn.setStreamName("detections")
        
        # Properties
        cam_rgb.setMaxDataSize(3 * 1920 * 1080)  # Maximum frame size
        
        # Network specific settings for person/phone detection
        detection_nn.setBlobPath(config.PERSON_MODEL_PATH)
        detection_nn.setConfidenceThreshold(0.20)  # Updated threshold
        detection_nn.setNumClasses(80)
        detection_nn.setCoordinateSize(4)
        detection_nn.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
        detection_nn.setAnchorMasks({
            "side52": [0, 1, 2],
            "side26": [3, 4, 5],
            "side13": [6, 7, 8]
        })
        detection_nn.setIouThreshold(0.5)
        detection_nn.setNumInferenceThreads(2)  # Updated to match new script
        detection_nn.setNumNCEPerInferenceThread(2)  # Optimize for RVC2
        detection_nn.setNumPoolFrames(2)  # Must match number of inference threads
        detection_nn.input.setBlocking(False)
        
        # Linking
        cam_rgb.out.link(detection_nn.input)
        detection_nn.passthrough.link(xout_rgb.input)
        detection_nn.out.link(xout_nn.input)
        
        # Set up seatbelt detection
        seatbelt_in = pipeline.create(dai.node.XLinkIn)
        seatbelt_in.setStreamName("seatbelt_in")
        
        seatbelt_nn = pipeline.create(dai.node.NeuralNetwork)
        seatbelt_nn.setBlobPath(config.SEATBELT_MODEL_PATH)
        seatbelt_nn.setNumPoolFrames(2)  # Updated to match inference threads
        seatbelt_nn.input.setBlocking(False)
        seatbelt_nn.input.setQueueSize(1)
        seatbelt_nn.setNumInferenceThreads(2)  # Updated to match new script
        seatbelt_nn.setNumNCEPerInferenceThread(2)  # Optimize for RVC2
        
        seatbelt_out = pipeline.create(dai.node.XLinkOut)
        seatbelt_out.setStreamName("seatbelt_out")
        
        # Link seatbelt nodes
        seatbelt_in.out.link(seatbelt_nn.input)
        seatbelt_nn.out.link(seatbelt_out.input)
        
        print("Pipeline created successfully")
        return pipeline
        
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        exit()