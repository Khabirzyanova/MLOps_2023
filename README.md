# Домашнее задание №1 

Реализовать API (REST либо процедуры gRPC), которое умеет:


1. Обучать ML-модель с возможностью настройки гиперпараметров. При этом гиперпараметры для разных моделей могут быть разные. Минимальное количество классов моделей доступных для обучения == 2.
2. Возвращать список доступных для обучения классов моделей
3. Возвращать предсказание конкретной модели (как следствие, система должна уметь хранить несколько обученных моделей)
4. Обучать заново и удалять уже обученные модели

*Оценка*:
* [4 балла] Работоспособность программы - то что ее можно запустить и она выполняет задачи, перечисленные в требованиях.
* [3 балла] Корректность и адекватность программы - корректная обработка ошибок, адекватный выбор структур классов, понятная документация (docstring-и адекатные здесь обязательны)
* [2 балла] Стиль кода - соблюдение стайлгайда. Буду проверять flake8 (не все ошибки на самом деле являются таковыми, но какие можно оставить – решать вам, насколько они критичны, списка нет☺)
* [1 балл] Swagger – Есть документация API (Swagger)
* [2 балла] – Реализация и REST API, и gRPC

### Создаем новое виртуальное окружение

У меня установлена miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), поэтому команды следующие:

`conda create --name [new_env] python=3.11`

`conda activate [new_env]`


### Устанавливаем необходимые пакеты
Файл requirements.txt содержит все необходимые зависимости. Для их загрузки воспользуйтесь командой:

`pip install -r requirements.txt`
