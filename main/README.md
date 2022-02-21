Цель: реализовать алгоритм хаотической оптимизации применительно для поиска оптимальных параметров в методе
опорных векторов и рассмотреть его применение к прогнозированию значений
временного ряда. Эффективность работы алгоритма проверялась на системе Лоренца.

За основу взята статья: Hu Yuxia , Zhang Hongtao, "Chaos Optimization Method of SVM Parameters
Selection for Chaotic Time Series Forecasting, Physics Procedia, 25 ( 2012 ), 588 –
594.

В ходе работы были реализованы следующие алгоритмы: 
  1. Для прогнозирования точек траектории использовался метод опорных векторов.
  2. Для вычисления значения размерности вложенного пространства - метод ближайших ложных соседей.
  3. Алгоритм хаотической оптимизации, основанный на результатах из статьи, для нахождения оптимальных параметров в методе опорных векторов.

Основной код работы представлен в файле Chaos_plot.ipynb