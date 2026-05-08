from typing import List, Tuple, Dict
from itertools import product, combinations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Solver:
    def __init__(self, f: str):
        """Принимает на вход функцию f, записанную в виде строки.
        
        Правила записи:
        * Переменные должны начинаться с буквы
        * Не должно быть переменных в верхнем и нижнем регистрах (например, a и A)

        Примеры:
        1. f = "a and (b or c)"
        2. f = "b and a and not(a and b)"
        3. f = "NOT((x1 OR x2) AND x3)"
        """
        # переводим функцию в нижний регистр
        self.f = f.lower() 
        logger.debug(f"f = {self.f}")

        # список переменных и их количество
        self.vars = self._get_vars()
        self.n = len(self.vars)

        # таблица истинности, словарь (x1, x2, ..., xn): f(x1, x2, ..., xn)
        self.truth_table = self._get_truth_table()

        # количество строк в карте Карно
        self.rows = 2 ** ((self.n - 2) // 2 + 1)
        logger.debug(f"rows = {self.rows}")

        # количество столбцов в карте Карно
        self.cols = 2 ** (((self.n - 2) // 2) + ((self.n - 2) % 2) + 1)
        logger.debug(f"cols = {self.cols}")

        # количество битов в строке (то есть количество цифр)
        self.x_bits = int(math.log2(self.rows))
        logger.debug(f"x_bits = {self.x_bits}")

        # количество битов в столбце
        self.y_bits = int(math.log2(self.cols))
        logger.debug(f"y_bits = {self.y_bits}")

        # карта Карно в виде таблицы
        self.karnaugh_map = self._get_karnaugh_map()

        # находим области, чтобы они покрывали все единицы в карте
        self.areas = self._get_areas()
        logger.debug(f"pairs = {self.areas}")

        # выводим минимизированную функцию
        self.output = self._get_output()
    
    def solve(self) -> str:
        """Возвращает строковое представление минимизированной булевой функции f."""
        return self.output
    
    def visualize_karnaugh_map(self) -> None:
        """Визуализирует карту Карно."""
        # переменные в строках
        row_vars = self.vars[:self.x_bits]
        # переменные в столбцах
        col_vars = self.vars[self.x_bits:]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')

        # создаем таблицу
        table = ax.table(
            cellText=self.karnaugh_map.values,
            colLabels=self.karnaugh_map.columns,
            rowLabels=self.karnaugh_map.index,
            cellLoc='center',
            loc='center'
        )

        table.scale(1, 2)
        table.set_fontsize(12)

        # добавляем подпись строк
        ax.text(-0.1, 0.5, " * ".join(row_vars), transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='center', rotation=90)

        # добавляем подпись столбцов
        ax.text(0.5, 0.8, " * ".join(col_vars), transform=ax.transAxes, 
                fontsize=14, fontweight='bold', ha='center')

        plt.show()
    
    def _get_vars(self) -> List[str]:
        """Возвращает список переменных в функции f."""
        strs = self.f.split(" ")
        reserved = set(["and", "or", "not", "(", ")"])
        vars = set()
        for s in strs:
            cur = ""
            for c in s:
                if c == ')' and len(cur) > 0:
                    vars.add(cur)
                    cur = ""
                if cur in reserved:
                    cur = ""
                cur += c
            if cur not in reserved and len(cur) > 0:
                vars.add(cur)

        res = sorted(list(vars)) # упорядочиваем переменные
        logger.debug(f"vars = {res}")
        return res
    
    def _get_truth_table(self) -> Dict[Tuple[int], int]:
        """Вычисляет таблицу истинности функции f."""
        result = {} # словарь вида {кортеж значений переменных: значение функции}

        # генерируем все возможные комбинации значений переменных
        for tuple in product([0, 1], repeat=self.n):
            d = dict(zip(self.vars, tuple)) # словарь вида {переменная: значение}
            result[tuple] = self._solve_boolean(**d) # добавляем результат

        logger.debug(f"truth_table = {result}")
        return result
    
    def _solve_boolean(self, **variables):
        """Вычисляет значение булевой функции f со значениями переменных из variables."""
        # преобразуем переменные в нижний регистр
        vars = {}
        for var, val in variables.items():
            vars[var.lower()] = val

        try:
            return eval(self.f, {}, variables)
        except NameError as e:
            return f"Ошибка: переменная {e} не определена"

    def _get_karnaugh_map(self) -> pd.DataFrame:
        """Вычисляет карту Карно."""
        
        data = [] # значения в таблице
        # названия строк
        index = [f"{self._to_gray(i):0{self.x_bits}b}" for i in range(self.rows)]
        # названия столбцов
        columns = [f"{self._to_gray(i):0{self.y_bits}b}" for i in range(self.cols)]
        
        # заполняем значения таблицы
        for row in index:
            for col in columns:
                data.append(self.truth_table[tuple(int(c) for c in row + col)])

        res = pd.DataFrame(data=np.array(data).reshape(self.rows, self.cols), 
                            index=index, 
                            columns=columns)
        logger.debug(f"karnaugh_map:\n{res}")
        return res

    def _to_gray(self, n):
        """Преобразует число в код Грея."""
        return n ^ (n >> 1)
    
    def _get_areas(self) -> List[List[tuple]]:
        """
        Возвращает список минимального количества областей, покрывающих единицы в карте Карно.
        Берёт области максимального размера.
        """
        # заполняем позиции с единицами в карте Карно
        valid_positions = set()
        for i in range(self.rows):
            for j in range(self.cols):
                if self.karnaugh_map.iloc[i, j] == 1:
                    valid_positions.add((i, j))
        
        # находим все области, покрывающие единицы
        all_areas = []
        size = self.rows * self.cols
        stop = False
        while size >= 2:
            for area in combinations(valid_positions, size):
                if not self._is_valid_area(area):
                    # если область невалидна, пропускаем
                    continue
                all_areas.append(area)
                seen = set()
                # проверяем, хватит ли нам всех областей для покрытия единиц
                for area in all_areas:
                    for pos in area:
                        seen.add(pos)
                if seen == valid_positions:
                    # если хватает, останавливаемся
                    stop = True
                    break
            if stop:
                break
            size //= 2
        
        # находим минимальные области
        for cnt in range(1, len(all_areas) + 1):
            for indices in combinations(range(len(all_areas)), cnt):
                min_areas = []
                seen = set()
                for i in indices:
                    area = all_areas[i]
                    min_areas.append(area)
                    for pos in area:
                        seen.add(pos)
                if seen == valid_positions:
                    return min_areas
    
    def _is_valid_area(self, coords):
        """Проверяет, является ли область валидной (прямоугольной)."""
        n = len(coords)

        # преобразуем координаты в код Грея
        indices = []
        for x, y in coords:
            gray_x = self._to_gray(x)
            gray_y = self._to_gray(y)
            indices.append((gray_x << self.y_bits) | gray_y)

        # находим маску изменяющихся бит
        bit_or = 0
        bit_and = indices[0]
        for idx in indices:
            bit_or |= idx
            bit_and &= idx
        
        diff_mask = bit_or ^ bit_and

        # для области размера 2^k должно меняться ровно k бит
        if bin(diff_mask).count('1') != n.bit_length() - 1:
            return False

        # проверка на полноту
        base = bit_and
        change_bits = [1 << i for i in range(self.x_bits + self.y_bits) 
                       if (diff_mask & (1 << i))]
        
        reconstructed = {base}
        for bit in change_bits:
            # удваиваем набор
            new_elements = {val | bit for val in reconstructed}
            reconstructed.update(new_elements)

        return set(indices) == reconstructed

    def _get_output(self) -> str:
        """Возвращает строковое представление минимизированой булевой функции f."""
        output = "f = "
        for area in self.areas:
            prod = "" # одна область - одно произведение
            str_repr = [] # строковое представление области в виде 0 и 1
            for x, y in area:
                s = self.karnaugh_map.index[x] + self.karnaugh_map.columns[y]
                str_repr.append(s)

            # находим стабильные переменные в области и добавляем их в ответ
            for i, digits in enumerate(zip(*str_repr)):
                # если есть и 0, и 1, то эту переменную можно игнорировать
                if len(set(digits)) != 1:
                    continue
                # остались либо все 0, либо все 1
                if digits[0] == "0":
                    prod += "not("
                prod += self.vars[i]
                if digits[0] == "0":
                    prod += ")"
                prod += " and "
            
            # нет стабильных переменных -> f = 1
            if prod == "":
                return "f = 1"

            # убираем последнее and и складываем со следующими произведениями
            output = output + prod[:-5] + " or "
            
        output = output[:-4] # убираем последнее or
        logger.debug(f"output = {output}")
        return output
    