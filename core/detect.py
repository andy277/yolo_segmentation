from ultralytics import YOLO


class Segmentation:
    def segment_detect(self, image_path):
        # Load a pretrained model
        model = YOLO("model/yolov8s-seg.pt")

        results = model(image_path)
        return results