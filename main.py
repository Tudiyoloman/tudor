import math

import math


class AI_Solution(object):
    def __init__(self):
        self.AI_test_euclidean_distance()
        self.AI_test_single_appeareance()
        self.AI_test_double_appearance()
        self.AI_test_majority_element()
        self.AI_test_kth_largest_element()
        self.test_scalar_product()
        self.test_generate_binary_to_n()
        self.test_matrix_most_ones_in_a_line()

        # print("All tests passed")

    def AI_euclidean_distance(self, point1, point2):
        """
        complexitate Teta(n) , n este dimensiunea in care se afla punctele
        """
        if len(point1) != len(point2):
            raise ValueError("Points must have the same number of dimensions")
        squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
        return math.sqrt(squared_distance)

    def AI_test_euclidean_distance(self):
        # Example usage:
        point1 = (1, 2, 3)
        point2 = (4, 5, 6)
        distance = self.AI_euclidean_distance(point1, point2)
        print("Euclidean distance:", distance)

    def AI_single_appearance(self, text):
        """
        complexitate timp O(n^2), pt fiecare cuvant se apeleaza functia count ce are complexitate Teta(n)
        complexitate spatiu O(n), n este numarul de cuvinte din text
        """
        words = text.split()
        return [word for word in words if words.count(word) == 1]

    def AI_test_single_appeareance(self):
        text = "ana are mere si mere"
        print(self.AI_single_appearance(text))

    def AI_find_duplicate(self, nums):
        """
        complexitate timp si spatiu O(n), n este numarul de elemente din nums
        """
        seen = set()
        for num in nums:
            if num in seen:
                return num
            seen.add(num)

    def AI_test_double_appearance(self):
        # Exemplu de utilizare:
        nums = [1, 2, 3, 4, 2]
        duplicate_value = self.AI_find_duplicate(nums)
        print("Valoarea care se repetă de două ori în șir este:", duplicate_value)

    def find_majority_element(self, nums):
        """
        Aici a gresit total nu are nicio legatura cu problema ceruta
        :param nums:
        :return:
        """
        candidate = None
        count = 0

        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1

        return candidate

    def AI_test_majority_element(self):
        # Exemplu de utilizare:
        nums = [2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]
        majority_element = self.find_majority_element(nums)
        print("Elementul majoritar în șir este:", majority_element)

    def kth_largest_element(self, nums, k):
        """
        complexitate timp O(n * k * logk)
        :param nums:
        :param k:
        :return:
        """

        # Sortăm primele k elemente
        sorted_subset = sorted(nums[:k], reverse=True)

        # Parcurgem restul elementelor din lista
        for num in nums[k:]:
            # Dacă numărul curent este mai mare decât cel mai mic număr din subsetul sortat,
            # îl înlocuim pe cel mai mic număr din subsetul sortat cu numărul curent și re-sortăm
            if num > sorted_subset[-1]:
                sorted_subset[-1] = num
                sorted_subset.sort(reverse=True)

        # Returnăm al k-lea cel mai mare element
        return sorted_subset[-1]

    def AI_test_kth_largest_element(self):
        nums = [7, 4, 6, 3, 9, 1]
        k = 2
        kth_largest = self.kth_largest_element(nums, k)
        print(f"Al {k}-lea cel mai mare element din șirul dat este:", kth_largest)

    def sparse_dot_product(self, vector1, vector2):
        """
        complexitate timp Teta(n) , n este dimensiunea in care se afla vectorii
        complexitate spatiu O(1)
        :param vector1:
        :param vector2:
        :return:
        """
        # Verificăm dacă lungimile vectorilor sunt aceleași
        if len(vector1) != len(vector2):
            raise ValueError("Vectorii trebuie să aibă aceeași lungime")

        # Inițializăm suma produselor
        dot_product = 0

        # Iterăm prin elementele vectorilor și calculăm suma produselor doar pentru elementele non-nule
        for i in range(len(vector1)):
            dot_product += vector1[i] * vector2[i]

        return dot_product

    def test_scalar_product(self):
        # Exemplu de utilizare:
        vector1 = [1, 0, 2, 0, 3]
        vector2 = [1, 2, 0, 3, 1]
        result = self.sparse_dot_product(vector1, vector2)
        print("Produsul scalar al vectorilor:", result)

    def generate_binary_numbers(self, n):
        """
        complexitate timp O(n * log2(n))
        :param n:
        :return:
        """
        # Iterăm de la 1 la n
        for i in range(1, n + 1):
            # Convertim numărul în reprezentarea sa binară și eliminăm prefixul '0b'
            binary_representation = bin(i)[2:]
            # Afisăm reprezentarea binară a numărului
            print(binary_representation)

    def test_generate_binary_to_n(self):
        # Testare cu n = 4
        n = 4
        self.generate_binary_numbers(n)

    def max_ones_row(self, matrix):
        max_ones_count = 0
        max_ones_row_index = -1

        for i, row in enumerate(matrix):
            ones_count = sum(row)
            if ones_count > max_ones_count:
                max_ones_count = ones_count
                max_ones_row_index = i

        return max_ones_row_index

    def test_matrix_most_ones_in_a_line(self):
        # Exemplu de utilizare:
        matrix = [
            [0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1]
        ]

        index = self.max_ones_row(matrix)
        print("Indexul liniei cu cele mai multe elemente de 1:", index)

class Solution(object):

    def __init__(self):
        print("Running tests for problems")

        self.test_euclidean_distance()
        self.test_single_appearance()
        self.test_double_appearance()
        self.test_majority_element()
        self.test_k_max_element()
        self.test_scalar_product()
        self.test_generate_binary_to_n()
        self.test_matrix_most_ones_in_a_line()

        print("All tests passed")

    def euclidean_distance(self, point1, point2):
        """
        Compute the euclidean distance between two points
        :param point1: list[int] representing coordinates of point1
        :param point2: list[int] representing coordinates of point2
        :return: The euclidean distance between point1 and point2
        :raises: ValueError: If the dimensions of the two points are not the same.
        complexitate Teta(n) , n este dimensiunea in care se afla punctele
        """

        if len(point1) != len(point2):
            raise ValueError("The two points must have the same number of coordinates")

        euclidean_distance = 0
        for index in range(len(point1)):
            euclidean_distance += (point2[index] - point1[index]) ** 2

        return math.sqrt(euclidean_distance)

    def improved_euclidean_distance(self, point1, point2):
        """
        Compute the euclidean distance between two points
        :param point1: list[int] representing coordinates of point1
        :param point2: list[int] representing coordinates of point2
        :return: The euclidean distance between point1 and point2
        :raises ValueError: If the dimensions of the two points are not the same.
        complexitate timp si spatiu Teta(n) , n este dimensiunea in care se afla punctele este o metoda mai "pythonica" de a face aceasta operatie, folosind list comprehension
        """

        if len(point1) != len(point2):
            raise ValueError("The two points must have the same number of coordinates")

        return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(point1, point2)))

    def test_euclidean_distance(self):
        try:
            self.euclidean_distance([1, 1], [4, 5, 0])
            assert False
        except ValueError:
            pass
        assert self.euclidean_distance([1, 1, 0], [4, 5, 0]) == 5.0
        assert self.euclidean_distance([1, 1], [4, 5]) == 5.0
        assert self.euclidean_distance([1, 1, 3], [4, 5, 6]) - 5.830951894845301 < 0.0001

    def single_appearance(self, text):
        """
        Se afiseaza cuvintele care apar o singura data in text
        :param text: String
        :return: List of strings
        complexitate timp O(n) , n este numarul de cuvinte din text
        complexitate spatiu O(n) , n este numarul de cuvinte din text
        DPDV al complexitatii nu exista o metoda mai eficienta de a face aceasta problema
        """
        map_appearance = {}
        for string in text.split():
            if string in map_appearance:
                map_appearance[string] += 1
            else:
                map_appearance[string] = 1

        list_of_single_appearance = []
        for key in map_appearance:
            if map_appearance[key] == 1:
                list_of_single_appearance.append(key)
        return list_of_single_appearance

    def test_single_appearance(self):
        assert self.single_appearance("ana are mere si mere") == ["ana", "are", "si"]
        assert self.single_appearance("ana are mere si pere si mere") == ["ana", "are", "pere"]
    def double_appearance(self, arr):
        """
        Se afiseaza singurul element care apare de doua ori in lista
        :param arr: list[int]
        :return: double_appeareance integer
        complexitate timp si spatiu Teta(n) , n este numarul de elemente din lista
        """
        map_appearance = {}
        for element in arr:
            if element in map_appearance:
                map_appearance[element] += 1
            else:
                map_appearance[element] = 1

        for key in map_appearance:
            if map_appearance[key] == 2:
                return key

    def improved_double_appearance(self, arr):
        """
        [1,2,3,2,4] - 12  das
        [1,2,3,4,5] - 15 das

        Se afiseaza singurul element care apare de doua ori in lista
        :param arr: list[int]
        :return: double_appeareance integer
        complexitate timp Teta(n) , n este numarul de elemente din lista
        dar reducem complexitatea spatiului la O(1)
        """
        suma_Gauss_n = len(arr) * (len(arr) + 1) // 2
        return len(arr) - (suma_Gauss_n - sum(arr))

    def test_double_appearance(self):
        assert self.double_appearance([1, 2, 3, 4, 5, 5, 6]) == 5
        assert self.double_appearance([1, 2, 3, 4, 5, 6, 6]) == 6

    def majority_element(self, arr):
        """
        Se afiseaza elementul care apare mai mult de n/2 ori in lista
        :param arr: list[int]
        :return: element majoritar sau None daca nu exista un element majoritar
        complexitate timp si spatiu Teta(n) , n este numarul de elemente din lista
         nu mai sunt improvizari de adus
        """
        map_appearance = {}
        for element in arr:
            if element in map_appearance:
                map_appearance[element] += 1
            else:
                map_appearance[element] = 1

        for key in map_appearance:
            if map_appearance[key] > len(arr) // 2:
                return key

    def test_majority_element(self):
        assert self.majority_element([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2
        assert self.majority_element([2, 8, 7, 2, 2, 5, 2, 3, 1, 2]) == None
        assert self.majority_element([1, 2, 2]) == 2

    def k_max_element(self, arr, k):
        """
        Se afiseaza al k element maxim din lista
        :param arr: list[int]
        :return: integer
        :raises: ValueError: Daca k este mai mare decat lungimea listei
        complexitate timp Teta(n*log2(n)) , n este numarul de elemente din lista
        complexitate spatiu Teta(n) , n este numarul de elemente din lista
        """
        if k > len(arr):
            raise ValueError("k is greater than the length of the list")
        sorted_arr = sorted(arr)
        return sorted_arr[-k]

    def test_k_max_element(self):
        assert self.k_max_element([7, 4, 8, 5, 2], 3) == 5
        assert self.k_max_element([7, 4, 8, 5, 2], 1) == 8
        try:
            self.k_max_element([7, 4, 8, 5, 2], 6)
            assert False
        except ValueError:
            pass

    def scalar_product(self, vector1, vector2):
        """
        Se calculeaza produsul scalar a doua vectori
        :param vector1: list[int]
        :param vector2: list[int]
        :return: integer
        :raise ValueError: Daca lungimile vectorilor nu sunt egale
        complexitate
        """
        if len(vector1) != len(vector2):
            raise ValueError("The two vectors must have the same length")
        sum = 0
        for index in range(len(vector1)):
            sum += vector1[index] * vector2[index]
        return sum

    def test_scalar_product(self):
        assert self.scalar_product([1, 2, 3], [4, 5, 6]) == 32
        assert self.scalar_product([1, 2, 0], [4, 5, 1]) == 14
        try:
            self.scalar_product([1, 2, 3], [4, 5])
            assert False
        except ValueError:
            pass

    def generate_binary_to_n(self, n):
        """
        Se genereaza toate numerele binare intre 1 si n
        :param n: integer
        :return: list of binary numbers
        complexitate timp O(n * log2(n))
        complexitate spatiu Teta(n)
        """
        return [bin(i)[2:] for i in range(1, n + 1)]

    def test_generate_binary_to_n(self):
        assert self.generate_binary_to_n(4) == ['1', '10', '11', '100']
        assert self.generate_binary_to_n(5) == ['1', '10', '11', '100', '101']
        assert self.generate_binary_to_n(6) == ['1', '10', '11', '100', '101', '110']

    def matrix_most_ones_in_a_line(self, matrix):
        """
        Se cauta linia cu cei mai multi de 1
        :param matrix: list[list[int]]
        :return: integer
        complexitate timp O(n*m), n este numarul de linii, m numarul de coloane
        """
        max_index = -1
        max_no_of_ones = -1
        for i in range(len(matrix)):
            current_no_of_ones = 0
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    current_no_of_ones += 1
            if current_no_of_ones > max_no_of_ones:
                max_no_of_ones = current_no_of_ones
                max_index = i

        return max_index

    def test_matrix_most_ones_in_a_line(self):
        matrix = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1]
        ]
        assert self.matrix_most_ones_in_a_line(matrix) == 2
        matrix = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1]
        ]
        assert self.matrix_most_ones_in_a_line(matrix) == 2


if __name__ == "__main__":
    solution = Solution()
    # print(solution.matrix_most_ones_in_a_line(matrix))
    # ai_solution = AI_Solution()
