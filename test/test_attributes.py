import unittest
from core.detect import Segmentation
import logging

logging.basicConfig(filename="logs/log.txt",
                    level=logging.INFO, filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',)
logger = logging.getLogger(__name__)


class MyTestCase(unittest.TestCase):
    def test_attributes(self):
        image_path = 'data/biker.jpg'

        segmentation = Segmentation()
        self.assertIsNotNone(segmentation)

        results = segmentation.segment_detect(image_path)[0]
        #Box attribute
        boxes = results.boxes
        self.assertIsNotNone(boxes)

        #original image
        orig_img = results.orig_img
        self.assertIsNotNone(orig_img)

        #original_shape
        orig_shape = results.orig_shape
        self.assertIsNotNone(orig_shape)

        #masks
        masks = results.masks
        self.assertIsNotNone(masks)

        #probs
        # probs = results.probs
        # self.assertIsNotNone(probs)
        #Probs is None

        #keypoints
        # keypoints = results.keypoints
        # self.assertIsNotNone(keypoints)
        #Keypoints is None

        #obb
        # obb = results.obb
        # self.assertIsNotNone(obb)
        #obb is None

        #speed
        speed = results.speed
        self.assertIsNotNone(speed)

        #names
        names = results.names
        self.assertIsNotNone(names)

        #path
        path = results.path
        self.assertIsNotNone(path)

        results.show()
        logger.info('Tests run')


if __name__ == '__main__':
    unittest.main()
