from ultralytics import YOLO


class Segmentation:
    def segment_detect(self, image_path):
        # Load a pretrained model
        model = YOLO("model/yolov8s-seg.pt")

        results = model(image_path)
        return results

    def segment_detect_call_predict(self, image_path, **kwargs):
        # Load a pretrained model
        model = YOLO("model/yolov8s-seg.pt")

        results = model.predict(image_path, **kwargs)
        return results