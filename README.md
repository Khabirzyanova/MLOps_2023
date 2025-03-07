# Домашнее задание №1 

Реализовать API (REST либо процедуры gRPC), которое умеет:


1. Обучать ML-модель с возможностью настройки гиперпараметров. При этом гиперпараметры для разных моделей могут быть разные. Минимальное количество классов моделей доступных для обучения == 2.
2. Возвращать список доступных для обучения классов моделей
3. Возвращать предсказание конкретной модели (как следствие, система должна уметь хранить несколько обученных моделей)
4. Обучать заново и удалять уже обученные модели

*Оценка*:
* [4 балла] Работоспособность программы - то что ее можно запустить и она выполняет задачи, перечисленные в требованиях.
* [3 балла] Корректность и адекватность программы - корректная обработка ошибок, адекватный выбор структур классов, понятная документация (docstring-и адекватные здесь обязательны)
* [2 балла] Стиль кода - соблюдение стайлгайда. Буду проверять flake8 (не все ошибки на самом деле являются таковыми, но какие можно оставить – решать вам, насколько они критичны, списка нет☺)
* [1 балл] Swagger – Есть документация API (Swagger)
* [2 балла] – Реализация и REST API, и gRPC

## Реализация с помощью FastAPI

https://github.com/testdrivenio/fastapi-ml

### Для запуска необходимо:

1. Склонировать репозиторий и перейти в папку hw1/fastapi-ml/:

```bash
$ git clone git@github.com:Khabirzyanova/MLOps_2023.git
$ cd hw1/fastapi-ml/
```

2. Создать новое виртуальное окружение (приведен пример для miniconda) и уставновить все необходимые зависимости из requirements.txt:
```bash
$ conda create --name [new_env] python=3.11
$ conda activate [new_env]
(new_env)$ pip install -r requirements.txt
```

3. Для запуска приложения на сервере:
```bash
(new_env)$ uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
```

### Создаем новое виртуальное окружение

У меня установлена miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), поэтому команды следующие:

`conda create --name [new_env] python=3.11`

`conda activate [new_env]`


### Устанавливаем необходимые пакеты
Файл requirements_fastapi.txt содержит все необходимые зависимости. Для их загрузки воспользуйтесь командой:

`pip install -r requirements_fastapi.txt`


## Реализация с помощью gRPC 

https://github.com/roboflow/deploy-models-with-grpc-pytorch-asyncio/tree/main

### Для запуска необходимо:

1. Склонировать репозиторий и перейти в папку hw1/grpc-ml/:

```bash
$ git clone git@github.com:Khabirzyanova/MLOps_2023.git
$ cd hw1/grpc-ml/
```

2. Создать новое виртуальное окружение (приведен пример для miniconda) и уставновить все необходимые зависимости из requirements.txt:
```bash
$ conda create --name [new_env] python=3.11
$ conda activate [new_env]
(new_env)$ pip install -r requirements.txt
```

### Создаем новое виртуальное окружение

У меня установлена miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), поэтому команды следующие:

`conda create --name [new_env] python=3.11`

`conda activate [new_env]`

### Устанавливаем необходимые пакеты
Файл requirements_grpc.txt содержит все необходимые зависимости. Для их загрузки воспользуйтесь командой:

`pip install -r requirements_grpc.txt`
