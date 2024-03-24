import unittest
import sys
import os

import numpy as np

# Add src/ and src/modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))

from src.data_generation.mot_data_generation import (
    generate_true_measurements,
    Object,
    FieldOfView,
)


class TestMotDataGenerator(unittest.TestCase):
    def test_measurements_from_two_objects_in_fov(self):
        t = 1
        obj_id1 = 1
        obj_id2 = 2
        objects_to_measure = [
            Object(
                pos=[0.1, 0],
                vel=[1, 1],
                t=t,
                delta_t=0.1,
                sigma=0.1,
                id=obj_id1,
            ),
            Object(
                pos=[1, 1],
                vel=[2, 2],
                t=t,
                delta_t=0.1,
                sigma=0.1,
                id=obj_id2,
            ),
        ]
        measurement_noises = np.zeros((2, 3))
        field_of_view = FieldOfView(
            min_range=0,
            max_range=10,
            max_range_rate=10,
            min_theta=-90,
            max_theta=90
        )

        true_measurements, obj_ids = generate_true_measurements(
            objects_to_measure,
            measurement_noises,
            t,
            field_of_view
        )

        self.assertEqual(len(true_measurements), 2)
        self.assertEqual(obj_ids[0], obj_id1)
        self.assertEqual(obj_ids[1], obj_id2)

    def test_measurements_from_one_object_outside_fov(self):
        t = 1
        obj_id1 = 1
        obj_id2 = 2
        objects_to_measure = [
            # Object inside the FOV
            Object(
                pos=[0.1, 0],
                vel=[1, 1],
                t=t,
                delta_t=0.1,
                sigma=0.1,
                id=obj_id1,
            ),
            # Object outside the FOV
            Object(
                pos=[100, 1],
                vel=[2, 2],
                t=t,
                delta_t=0.1,
                sigma=0.1,
                id=obj_id2,
            ),
        ]
        measurement_noises = np.zeros((2, 3))
        field_of_view = FieldOfView(
            min_range=0,
            max_range=10,
            max_range_rate=10,
            min_theta=-90,
            max_theta=90
        )

        true_measurements, obj_ids = generate_true_measurements(
            objects_to_measure,
            measurement_noises,
            t,
            field_of_view
        )

        self.assertEqual(len(true_measurements), 1)
        self.assertEqual(len(obj_ids), 1)


if __name__ == '__main__':
    unittest.main()
