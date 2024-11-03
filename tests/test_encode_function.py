import unittest

from source.encode import encode_number, encode_numbers
from source.numeric_representation import EmbeddingClasses


class TestEncodeFunction(unittest.TestCase):
    def test_encode_numbers(self):
        input = ["124", 124, 12.4, 0, 1.24, "124 one hunder twenty four"]

        for element in EmbeddingClasses:
            encode_numbers(input=input, embedding_type=element.value)

    def test_encode_number(self):
        input = "124"

        for element in EmbeddingClasses:
            encode_number(input=input, embedding_type=element.value)

    def test_encode_numbers_2(self):
        numerical_input: list[str | int | float] = [3.14, 10]
        for element in EmbeddingClasses:
            encode_numbers(input=numerical_input, embedding_type=element.value)

        string_input: list[str | int | float] = [
            "Rated 5 stars",
            "The product costs 23 dollars",
        ]
        for element in EmbeddingClasses:
            encode_numbers(input=string_input, embedding_type=element.value)

    def test_encode_number_2(self):
        for input in [
            3.14,
            10,
            "Rated 5 stars",
            "The product costs 23 dollars",
        ]:
            for element in EmbeddingClasses:
                encode_number(input=input, embedding_type=element.value)


if __name__ == "__main__":
    unittest.main()
