if __name__ == '__main__':
    import ultralytics
    from ultralytics import YOLO

    # Load the model
    model = YOLO('yolov8n-pose.pt')

    # Train the model
    results = model.train(data="./pose.yaml", epochs=100, imgsz=640)
