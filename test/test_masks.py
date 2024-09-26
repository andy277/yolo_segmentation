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
        self.assertIsNotNone(results)

        masks = results.masks
        self.assertIsNotNone(masks)

        data_as_numpy = masks.data.numpy()
        self.assertIsNotNone(data_as_numpy)

        xy = masks.xy
        self.assertIsNotNone(xy)

        for i in range(len(xy)):
            self.assertIsNotNone(xy[i])
            print(f"Object {i+1}", xy[i])

        # results.show()
        results.save('results/results.jpg')

        logger.info('Tests run')


if __name__ == '__main__':
    unittest.main()
