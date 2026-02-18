#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа для построения графика по точкам с поддержкой Matplotlib и Bokeh
и автоматическим форматированием кода с помощью Black.
"""

import sys
import os
import argparse
import logging
import subprocess
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Point:
    """Класс для представления точки с координатами x и y"""
    x: float
    y: float

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def to_tuple(self) -> Tuple[float, float]:
        """Преобразование в кортеж"""
        return (self.x, self.y)


class PointReader:
    """Класс для чтения точек из различных источников"""

    @staticmethod
    def from_console() -> List[Point]:
        """Чтение точек из консоли"""
        print("\n" + "=" * 50)
        print("ВВОД ТОЧЕК С КЛАВИАТУРЫ")
        print("=" * 50)
        print("Введите координаты точек (x y), по одной точке на строку.")
        print("Для завершения ввода введите пустую строку:\n")

        points = []
        line_number = 1

        while True:
            try:
                line = input(f"Точка {line_number:2d}> ").strip()

                if not line:
                    if line_number > 1:  # Хотя бы одна точка введена
                        break
                    else:
                        continue

                # Разделяем строку и преобразуем в числа
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError("Требуется два числа")

                x, y = float(parts[0]), float(parts[1])
                points.append(Point(x, y))
                line_number += 1

            except ValueError as e:
                logger.error(f"Ошибка в формате данных: {e}")
                print(
                    "   Правильный формат: два числа, разделенных пробелом (например: 3.14 2.5)"
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nВвод прерван пользователем")
                break

        return points

    @staticmethod
    def from_file(filename: Union[str, Path]) -> Optional[List[Point]]:
        """Чтение точек из файла"""
        points = []
        file_path = Path(filename)

        if not file_path.exists():
            logger.error(f"Файл '{filename}' не найден")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Пропускаем пустые строки и комментарии
                    if not line or line.startswith("#"):
                        continue

                    try:
                        parts = line.split()
                        if len(parts) != 2:
                            logger.warning(
                                f"Строка {line_num} пропущена: '{line}' - ожидалось два числа"
                            )
                            continue

                        x, y = float(parts[0]), float(parts[1])
                        points.append(Point(x, y))
                    except ValueError as e:
                        logger.warning(f"Строка {line_num} пропущена: '{line}' - {e}")

        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {e}")
            return None

        return points


class PointWriter:
    """Класс для записи точек в файл"""

    @staticmethod
    def to_file(points: List[Point], filename: Union[str, Path]) -> bool:
        """Сохранение точек в файл"""
        try:
            file_path = Path(filename)

            # Создаем директорию, если она не существует
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Точки для построения графика\n")
                f.write("# Формат: x y\n")
                f.write(f"# Всего точек: {len(points)}\n")
                f.write("# " + "=" * 40 + "\n")

                for i, point in enumerate(points, 1):
                    f.write(f"{point.x:.6f} {point.y:.6f}")
                    if i < len(points):
                        f.write("\n")

            logger.info(f"Данные сохранены в файл: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении в файл: {e}")
            return False


class PlotFactory:
    """Фабрика для создания графиков с использованием различных библиотек"""

    @staticmethod
    def create_plot(
        points: List[Point],
        title: str = "График по точкам",
        library: str = "matplotlib",
        save_path: Optional[str] = None,
    ) -> None:
        """Создание графика с указанной библиотекой"""

        if not points:
            logger.warning("Нет данных для построения графика")
            return

        if library == "matplotlib":
            PlotFactory._plot_with_matplotlib(points, title, save_path)
        elif library == "bokeh":
            PlotFactory._plot_with_bokeh(points, title, save_path)
        else:
            logger.error(f"Неизвестная библиотека: {library}")
            logger.info("Доступные библиотеки: matplotlib, bokeh")

    @staticmethod
    def _plot_with_matplotlib(
        points: List[Point], title: str, save_path: Optional[str]
    ) -> None:
        """Построение графика с помощью Matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            logger.error(f"Библиотека matplotlib не установлена: {e}")
            logger.info("Установите: pip install matplotlib numpy")
            return

        # Создание фигуры с высоким качеством
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        # Построение основной линии с точками
        (line,) = ax.plot(
            x_coords,
            y_coords,
            "o-",
            color="#2E86AB",
            linewidth=2.5,
            markersize=8,
            markerfacecolor="#A23B72",
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Точки",
            zorder=3,
        )

        # Добавление подписей для каждой точки
        for i, point in enumerate(points):
            # Определяем смещение для подписи в зависимости от положения точки
            if i == 0:
                offset = (10, 10)
            elif i == len(points) - 1:
                offset = (-10, -10)
            else:
                offset = (0, 15)

            # Форматируем подпись в зависимости от величины чисел
            if abs(point.x) < 10 and abs(point.y) < 10:
                label = f"({point.x:.2f}, {point.y:.2f})"
            else:
                label = f"({point.x:.1f}, {point.y:.1f})"

            ax.annotate(
                label,
                xy=(point.x, point.y),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="none"
                ),
                zorder=4,
            )

        # Добавление интерполяции для более гладкой кривой (если точек > 3)
        if len(points) > 3:
            try:
                from scipy import interpolate

                # Параметрическая интерполяция
                t = np.arange(len(points))
                ti = np.linspace(0, len(points) - 1, 300)

                cs_x = interpolate.CubicSpline(t, x_coords)
                cs_y = interpolate.CubicSpline(t, y_coords)

                xi = cs_x(ti)
                yi = cs_y(ti)

                ax.plot(
                    xi,
                    yi,
                    "--",
                    color="#86B3D1",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Интерполяция",
                    zorder=2,
                )
            except ImportError:
                pass

        # Настройка сетки
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        # Настройка осей
        ax.set_xlabel("Ось X", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ось Y", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Добавление легенды
        ax.legend(loc="best", framealpha=0.9)

        # Автоматическое масштабирование с небольшим отступом
        ax.margins(x=0.1, y=0.15)

        # Настройка формата чисел на осях
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        # Добавление информации о количестве точек
        textstr = f"Всего точек: {len(points)}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        # Сохранение или отображение
        if save_path:
            try:
                # Определяем формат по расширению файла
                file_ext = Path(save_path).suffix.lower()

                if file_ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg"]:
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    logger.info(f"График сохранен: {save_path}")
                else:
                    # Если расширение не указано или не поддерживается, добавляем .png
                    save_path_png = str(Path(save_path).with_suffix(".png"))
                    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
                    logger.info(f"График сохранен: {save_path_png}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении графика: {e}")
                plt.show()
        else:
            plt.show()

    @staticmethod
    def _plot_with_bokeh(
        points: List[Point], title: str, save_path: Optional[str]
    ) -> None:
        """Построение графика с помощью Bokeh"""
        try:
            from bokeh.plotting import figure, show, output_file
            from bokeh.models import HoverTool, LabelSet, ColumnDataSource
            from bokeh.io import save
        except ImportError as e:
            logger.error(f"Библиотека bokeh не установлена: {e}")
            logger.info("Установите: pip install bokeh")
            return

        # Подготовка данных
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        # Создание источника данных
        source = ColumnDataSource(
            data={
                "x": x_coords,
                "y": y_coords,
                "labels": [f"({p.x:.2f}, {p.y:.2f})" for p in points],
                "index": list(range(len(points))),
                "x_str": [f"{p.x:.3f}" for p in points],
                "y_str": [f"{p.y:.3f}" for p in points],
            }
        )

        # Создание фигуры
        p = figure(
            title=title,
            x_axis_label="Ось X",
            y_axis_label="Ось Y",
            width=1000,
            height=600,
            toolbar_location="above",
            tools="pan,box_zoom,reset,save,wheel_zoom,crosshair",
            active_scroll="wheel_zoom",
        )

        # Настройка внешнего вида
        p.title.text_font_size = "16pt"
        p.title.text_font_style = "bold"
        p.title.align = "center"

        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "10pt"
        p.yaxis.major_label_text_font_size = "10pt"

        p.background_fill_color = "#f8f9fa"
        p.border_fill_color = "white"

        # Добавление линии с точками
        line_renderer = p.line(
            "x", "y", source=source, line_width=3, color="#2E86AB", legend_label="Линия"
        )

        circle_renderer = p.circle(
            "x",
            "y",
            source=source,
            size=12,
            color="#A23B72",
            fill_alpha=0.8,
            line_color="white",
            line_width=2,
            legend_label="Точки",
        )

        # Добавление подписей для точек
        labels = LabelSet(
            x="x",
            y="y",
            text="labels",
            source=source,
            text_align="center",
            text_baseline="bottom",
            y_offset=10,
            text_font_size="9pt",
            text_color="#333333",
            background_fill_color="yellow",
            background_fill_alpha=0.7,
            border_line_color=None,
            padding=3,
        )
        p.add_layout(labels)

        # Добавление интерактивных подсказок
        hover = HoverTool(
            renderers=[circle_renderer],
            tooltips=[
                ("Точка", "@index"),
                ("X", "@x{0.000}"),
                ("Y", "@y{0.000}"),
                ("Координаты", "@labels"),
            ],
            mode="mouse",
        )
        p.add_tools(hover)

        # Настройка сетки
        p.grid.grid_line_color = "#e0e0e0"
        p.grid.grid_line_alpha = 0.5
        p.grid.grid_line_dash = [4, 4]

        # Настройка легенды
        p.legend.location = "top_left"
        p.legend.label_text_font_size = "10pt"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_width = 1
        p.legend.border_line_color = "#cccccc"

        # Добавление минимальных и максимальных значений
        if points:
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)

            # Добавление линий минимумов/максимумов
            p.hspan(
                y=[min_y, max_y], line_color="green", line_alpha=0.2, line_dash="dashed"
            )

            p.vspan(x=[min_x, max_x], line_color="red", line_alpha=0.2, line_dash="dashed")

        # Сохранение или отображение
        if save_path:
            try:
                # Убеждаемся, что файл имеет правильное расширение
                if not save_path.endswith((".html", ".htm")):
                    save_path += ".html"

                output_file(save_path, title=title)
                save(p)
                logger.info(f"Интерактивный график сохранен: {save_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении графика: {e}")
                show(p)
        else:
            show(p)


class BlackFormatter:
    """Класс для форматирования кода с помощью Black"""

    @staticmethod
    def check_installation() -> bool:
        """Проверка установки Black"""
        try:
            import black

            return True
        except ImportError:
            return False

    @staticmethod
    def format_file(filepath: Union[str, Path], check_only: bool = False) -> bool:
        """
        Форматирование файла с помощью Black

        Args:
            filepath: Путь к файлу для форматирования
            check_only: Только проверить, не изменяя файл

        Returns:
            True если форматирование выполнено успешно или ошибок нет
        """
        file_path = Path(filepath)

        if not file_path.exists():
            logger.error(f"Файл не найден: {filepath}")
            return False

        if not BlackFormatter.check_installation():
            logger.error("Black не установлен. Установите: pip install black")
            return False

        try:
            import black

            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Настройки форматирования
            mode = black.Mode(
                line_length=100,
                string_normalization=True,
                is_pyi=False,
            )

            if check_only:
                # Проверка без изменений
                try:
                    black.format_str(source, mode=mode)
                    logger.info("✓ Black: форматирование в порядке")
                    return True
                except Exception as e:
                    logger.warning(f"✗ Black: требуется форматирование - {e}")
                    return False
            else:
                # Форматирование с сохранением
                formatted = black.format_str(source, mode=mode)

                if formatted != source:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(formatted)
                    logger.info("✓ Black: файл отформатирован")
                else:
                    logger.info("✓ Black: форматирование не требуется")

                return True

        except Exception as e:
            logger.error(f"Ошибка при форматировании Black: {e}")
            return False

    @staticmethod
    def format_directory(
        directory: Union[str, Path], extensions: List[str] = None, check_only: bool = False
    ) -> Tuple[int, int]:
        """
        Форматирование всех Python файлов в директории

        Args:
            directory: Путь к директории
            extensions: Список расширений файлов
            check_only: Только проверить, не изменяя файлы

        Returns:
            Кортеж (количество обработанных файлов, количество измененных файлов)
        """
        if extensions is None:
            extensions = [".py"]

        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Директория не найдена: {directory}")
            return (0, 0)

        processed = 0
        changed = 0

        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    logger.info(f"Обработка: {file_path}")
                    processed += 1

                    if not check_only:
                        # Сохраняем оригинал для сравнения
                        with open(file_path, "r", encoding="utf-8") as f:
                            original = f.read()

                        if BlackFormatter.format_file(file_path, check_only=False):
                            with open(file_path, "r", encoding="utf-8") as f:
                                new = f.read()
                            if original != new:
                                changed += 1
                    else:
                        if not BlackFormatter.format_file(file_path, check_only=True):
                            changed += 1

        return (processed, changed)

    @staticmethod
    def format_string(code: str) -> str:
        """Форматирование строки кода"""
        try:
            import black

            mode = black.Mode(line_length=100)
            return black.format_str(code, mode=mode)
        except ImportError:
            return code
        except Exception as e:
            logger.error(f"Ошибка форматирования строки: {e}")
            return code

    @staticmethod
    def install() -> bool:
        """Установка Black"""
        try:
            import black

            logger.info("Black уже установлен")
            return True
        except ImportError:
            logger.info("Установка Black...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "black"])
                logger.info("Black успешно установлен")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка установки Black: {e}")
                return False

    @staticmethod
    def get_version() -> str:
        """Получение версии Black"""
        try:
            import black

            return black.__version__
        except ImportError:
            return "Не установлен"
        except AttributeError:
            return "Неизвестно"


def setup_argparse() -> argparse.ArgumentParser:
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Программа для построения графика по точкам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s -i points.txt -o graph.png -t "Мой график"
  %(prog)s -i points.txt -d saved_points.txt -l bokeh
  %(prog)s --black-format plot_points.py
  %(prog)s --black-check plot_points.py
  %(prog)s --black-install
  %(prog)s --black-dir ./src --black-check
        """,
    )

    # Основные аргументы
    parser.add_argument(
        "-i", "--input", type=str, help="Входной файл с точками (если не указан, ввод с консоли)"
    )
    parser.add_argument("-o", "--output", type=str, help="Выходной файл для сохранения графика")
    parser.add_argument("-d", "--data", type=str, help="Сохранить введенные точки в файл")
    parser.add_argument(
        "-t", "--title", type=str, default="График по точкам", help="Заголовок графика"
    )

    # Выбор библиотеки для визуализации
    parser.add_argument(
        "-l",
        "--library",
        type=str,
        choices=["matplotlib", "bokeh"],
        default="matplotlib",
        help="Библиотека для построения графика",
    )

    # Аргументы для Black
    black_group = parser.add_argument_group("Black formatter options")
    black_group.add_argument(
        "--black-format", type=str, metavar="FILE", help="Отформатировать файл с помощью Black"
    )
    black_group.add_argument(
        "--black-check",
        type=str,
        metavar="FILE",
        help="Проверить форматирование файла (без изменений)",
    )
    black_group.add_argument(
        "--black-dir", type=str, metavar="DIR", help="Отформатировать все Python файлы в директории"
    )
    black_group.add_argument(
        "--black-install", action="store_true", help="Установить Black"
    )
    black_group.add_argument(
        "--black-version", action="store_true", help="Показать версию Black"
    )

    # Дополнительные опции
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод информации")
    parser.add_argument(
        "--version", action="version", version="%(prog)s 2.0 (Black formatter)"
    )

    return parser


def print_banner() -> None:
    """Вывод приветственного баннера"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║     ПРОГРАММА ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ ПО ТОЧКАМ        ║
    ║         Поддержка Matplotlib и Bokeh                    ║
    ║         Форматирование кода с помощью Black            ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main() -> None:
    """Главная функция программы"""
    parser = setup_argparse()
    args = parser.parse_args()

    # Настройка уровня логирования
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Проверка версии Black
    if args.black_version:
        version = BlackFormatter.get_version()
        print(f"Black version: {version}")
        sys.exit(0)

    # Установка Black
    if args.black_install:
        success = BlackFormatter.install()
        sys.exit(0 if success else 1)

    # Форматирование файла
    if args.black_format:
        success = BlackFormatter.format_file(args.black_format, check_only=False)
        sys.exit(0 if success else 1)

    # Проверка файла
    if args.black_check:
        success = BlackFormatter.format_file(args.black_check, check_only=True)
        sys.exit(0 if success else 1)

    # Форматирование директории
    if args.black_dir:
        processed, changed = BlackFormatter.format_directory(
            args.black_dir, check_only=(args.black_check is not None)
        )
        print(f"Обработано файлов: {processed}")
        print(f"Требуется форматирование: {changed}")
        sys.exit(0 if changed == 0 else 1)

    # Вывод баннера
    print_banner()

    # Основная логика программы
    try:
        # Чтение данных
        if args.input:
            logger.info(f"Чтение данных из файла: {args.input}")
            points = PointReader.from_file(args.input)
            if points is None:
                sys.exit(1)
        else:
            logger.info("Режим консольного ввода")
            points = PointReader.from_console()

        if not points:
            logger.error("Нет данных для обработки")
            sys.exit(1)

        logger.info(f"Прочитано {len(points)} точек")

        # Вывод прочитанных точек для проверки
        if args.verbose:
            print("\nПрочитанные точки:")
            for i, point in enumerate(points, 1):
                print(f"  {i:2d}. {point}")

        # Сохранение данных
        if args.data:
            PointWriter.to_file(points, args.data)

        # Построение графика
        logger.info(f"Построение графика с использованием {args.library}")
        PlotFactory.create_plot(points, args.title, args.library, args.output)

    except KeyboardInterrupt:
        logger.info("\nПрограмма прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()