import unittest

from unique_person_counter import UniquePersonCounter


class UniquePersonCounterTests(unittest.TestCase):
    def test_counts_new_confirmed_person_once(self):
        counter = UniquePersonCounter()

        self.assertEqual(counter.update(7, "CONFIRMED"), 1)
        self.assertEqual(counter.update(7, "CONFIRMED"), 1)
        self.assertEqual(counter.update(9, "CONFIRMED"), 2)

    def test_ignores_tentative_or_unconfirmed_persons(self):
        counter = UniquePersonCounter()

        self.assertEqual(counter.update(11, "TENTATIVE"), 0)
        self.assertEqual(counter.update(11, "CONFIRMED"), 1)
        self.assertEqual(counter.update(11, "TRACK_LOCKED"), 1)


if __name__ == "__main__":
    unittest.main()
