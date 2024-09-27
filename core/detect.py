from ultralytics import YOLO


class Segmentation:
    """
    Predict image by calling model(image_path)

    Args:
        image_path (str): Path to image file to predict

    Returns:
        Results Object: Object retuns by YOLO model
    """
    def segment_detect(self, image_path):
        # Load a pretrained model
        model = YOLO("model/yolov8s-seg.pt")

        results = model(image_path)
        return results

    """
       Predict image by calling model.predict(image_path)

       Args:
           image_path (str): Path to image file to predict
           **kwargs: refer to YOLO arguments

       Returns:
           Results Object: Object retuns by YOLO model
       """
    def segment_detect_call_predict(self, image_path, **kwargs):
        # Load a pretrained model
        model = YOLO("model/yolov8s-seg.pt")

        results = model.predict(image_path, **kwargs)
        return results