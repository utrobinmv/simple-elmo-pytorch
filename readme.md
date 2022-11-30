Реализация нейросетевой архитектуры ELMO

Данный репозиторий, является уменьшенной версий реализации ELMO из [AllenNLP](https://github.com/allenai/allennlp).



Особенность данной библиотеки в том, что она практически не использует зависимостей кроме pytorch

Поддерживает загрузку обученных моделей ELMO с сайта [RusVectores](https://rusvectores.org/ru/models/).



## Инструкция по использованию

Установка

```bash
python setup.py install
```

или

```bash
pip install simple_elmo_pytorch
```



Инициализация

```python
from simple_elmo_pytorch import Elmo
from simple_elmo_pytorch import batch_to_ids

from simple_elmo_pytorch import ElmoVectorizer
```

Загрузка модели

```python
model = Elmo(options_file='models/elmo/196/options.json',
            weight_file='models/elmo/196/model.hdf5',
            num_output_representations = 2
            )
```

Использование Вариант 1:

можно использовать как, прямое получения выхода с модели

```python
text_str_1 = 'Привет моя строка как дела ура'
text_str_2 = 'Привет моя вторая строка как дела ура'
text_str_3 = 'Привет моя третья строка'

sentence_ids = batch_to_ids([text_str_1.split(),text_str_2.split(),text_str_3.split()])

model.eval()
out = model(sentence_ids)
```

Использование Вариант 2:

Реализованы процедуры, работающие аналогично библиотеке [simple_elmo](https://github.com/ltgoslo/simple_elmo)

- get_elmo_vectors(SENTENCES) - для получения эмбединга каждого слова
- get_elmo_vector_average(SENTENCES)`- для получение эмбединга всей последовательности, как среднее всех слов

```python
vk = ElmoVectorizer(model, batch_size = 2, device='cpu')

text_str_1 = 'Привет моя строка как дела ура'
text_str_2 = 'Привет моя вторая строка как дела ура'
text_str_3 = 'Привет моя третья строка'

v = vk.get_elmo_vector_average([text_str_1.split(' '), text_str_2.split(' '), text_str_3.split(' ')], warmup=True)

v = vk.get_elmo_vectors([text_str_1.split(' '), text_str_2.split(' '), text_str_3.split(' ')], warmup=True)

```

