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

        # карта Карно в виде таблицы
        self.karnaugh_map = self._get_karnaugh_map()

        # находим пары элементов, чтобы они покрывали все единицы в карте
        self.pairs = self._get_pairs()

        # выводим минимизированную функцию
        self.output = self._get_output()
    
    def solve(self) -> str:
        """Возвращает строковое представление минимизированной булевой функции f."""
        return self.output
    
    def visualize_karnaugh_map(self) -> None:
        """Визуализирует карту Карно."""
        # переменные в строках
        row_vars = self.vars[:int(math.log2(self.rows))]
        # переменные в столбцах
        col_vars = self.vars[int(math.log2(self.rows)):]

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
        def gray_code(n, digits):
            """Заполняет n чисел согласно коду Грея и возвращает строковое представление из digits элементов."""
            result = []
            for i in range(n):
                # формула кода Грея: G = B XOR (B >> 1), где B - двоичное число
                gray = i ^ (i >> 1)
                gray = f"{gray:0{digits}b}" # преобразуем в строковый вид с нулями слева
                result.append(gray) # добавляем строку
            
            return result
        
        data = [] # значения в таблице
        index = gray_code(self.rows, int(math.log2(self.rows))) # названия строк
        columns = gray_code(self.cols, int(math.log2(self.cols))) # названия столбцов
        
        # заполняем значения таблицы
        for row in index:
            for col in columns:
                data.append(self.truth_table[tuple(int(c) for c in row + col)])

        res = pd.DataFrame(data=np.array(data).reshape(self.rows, self.cols), 
                            index=index, 
                            columns=columns)
        logger.debug("karnaugh_map")
        logger.debug(res)
        return res
    
    def _get_pairs(self) -> List[List[tuple]]:
        """Возвращает минимально возможный список пар переменных, состоящих из 1 в карте Карно."""
        # заполняем позиции с единицами в карте Карно
        valid_positions = set()
        for i in range(self.rows):
            for j in range(self.cols):
                if self.karnaugh_map.iloc[i, j] == 1:
                    valid_positions.add((i, j))
        
        # группируем по 2 элемента по вертикали или горизонтали (не всегда эффективно)
        all_pairs = []
        for pos in valid_positions:
            for other_pos in valid_positions:
                # если позиции не совпадают, то проверяем, что они находятся в одном ряду или столбце
                if pos != other_pos and (pos[0] == other_pos[0] or pos[1] == other_pos[1]):
                    # не добавляем дубликаты
                    if (other_pos, pos) not in all_pairs:
                        all_pairs.append((pos, other_pos))
        
        
        # находим минимальные пары
        for cnt in range(1, len(all_pairs)):
            for indices in combinations(range(len(all_pairs)), cnt):
                result = []
                seen = set()
                for i in indices:
                    pair = all_pairs[i]
                    result.append(pair)
                    for pos in pair:
                        seen.add(pos)
                if seen == valid_positions:
                    return result

    def _get_output(self) -> str:
        """Возвращает строковое представление минимизированой булевой функции f."""
        output = "f = "
        for pair in self.pairs:
            prod = ""
            str_repr1 = self.karnaugh_map.index[pair[0][0]] + self.karnaugh_map.columns[pair[0][1]]
            str_repr2 = self.karnaugh_map.index[pair[1][0]] + self.karnaugh_map.columns[pair[1][1]]
            for i, (c1, c2) in enumerate(zip(str_repr1, str_repr2)):
                if c1 == c2:
                    if c1 == "0":
                        prod += "not("
                    prod += self.vars[i]
                    if c1 == "0":
                        prod += ")"
                    prod += " and "
            output = output + prod[:-5] + " or "
        output = output[:-4]
        logger.debug(f"output = {output}")
        return output
    

sol = Solver("not(x1) and x2 and x3 or x1 and x2 and x3 or x1 and not(x2) and not(x3) or x1 and not(x2) and x3")
sol = Solver("not(x1) and not(x2) and not(x3) or not(x1) and not(x2) and x3 or not(x1) and x2 and x3 or x1 and not(x2) and not(x3) or x1 and not(x2) and not(x3) or x1 and not(x2) and x3 or x1 and x2 and x3")
out = sol.output
sec = Solver(out[4:])