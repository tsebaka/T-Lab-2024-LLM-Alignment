# T-Lab-2024-LLM-Alignment

![meme](https://github.com/tsebaka/T-Lab-2024-LLM-Alignment/blob/main/meme/memes.jpg?raw=true)

## Mini-batch итогов
- Изначально я взял пары и обучил на них reward модельку (одной эпохи было достаточно чтобы выбить 80% качества на test выборке с sign трешхолдом для класса (это я так проверил норм она или нет)) и на всякий случай я закинул её на яндекс диск (model, conf)
- Потом я реализовал метод reinforce (sorry trl в вашем коде я немного запутался)
- Потом добавил метод WARP (а именно EMA, SLERP, LITI)
- После этого я заметил, что то, что генерит sft-шка это почти всегда GOOD MOVIE GOOD MOVIE (без сильного подбора beta и T), и поэтому я подумал что нужно добавить метод RLOO чтобы дисперсия была меньше и модель лучше обучалась на k разных развитий генерации (взял отсюда [Back to Basics](https://arxiv.org/pdf/2402.14740)). Что удивительно (типа реально заработало) у меня результаты reward'а увеличились и ответы были более разнообразные, а не просто FUCK IT'S A GREAT MOVIE.

## Ultra-mini-batch того что я понял
### В общем
- Результаты были понятны (увеличиваешь бету, модель хуже генерит хорошие комменты, перетягивая всё на kl loss, соответственно avg_reward_before &asymp; avg_reward_after и avg_kl_before &asymp; avg_kl_after)
- Кол-во параметров T чем больше, тем лучше модель будет обучаться (вообще логично потому что по факту 100 промтов - маловато), но при этом увеличивается KL Loss, и есть вероятность, что FUCK GOOD HELLO GREAT FILM вернётся, поэтому нужно понять следующее
### EMA
- При росте T, кол-во динамических опорных точек должно рости, иначе всё сколлапсится => чем больше M, тем больше опорных точек в степах, но тем больше нужно данных для обучение (иначе ниче не заработает)
- В коде всё написано для двух поэтому хрен так легко это поменяешь...
### SLERP 
- Как будто очевидно что должно работать лучше LERP, ведь это более гладкий переход к сумме двух векторов. Думал добавить его продолжение (метод SQUAD), но я в этих кватернионах ваще ниче не понял :(
### LITI 
- ну ок...
### k параметр в RLOO
- типа как параметр k в бим серче, у меня становилось лучше и стабильнее результаты
- Так же я юзал кл лосс по всем k сэмплам, хотя можно было и просто пикать рандомный элемент и считать его аутпут с sft base и текущей

## Не понял
- Не получилось реализовать если я подаю batch сэмплов в forward, ниче не обучалось
- о каком batch size шла речь в тестовом задании поэтому возможно зашумленность моих результатов связана с этим (пушто я не понял по 100 итераций по батчу или по 100 итераций батчей)


## Резы

### WARP

Во всех табличках средний reward before это ~ 0.3 +- 0.2, а примерный kl before это что близко к нулю (логично): -1.46e-7
beta=0.01
| T   | reward      | kl          | нормальность генераций на мой глаз  |
|-----|-------------|-------------|-------------------------------------|
| 100 | 2.4         | 9.1         | Норм генерации но циклится (я надеюсь что это приколы gpt2)       | 
| 500 | 2.5         | 5.8         | I was so excited to see this movie чередуясь с тем что сверху     |
| 1000| 3.2         | 16.56       | Тут явно разошлась потому что вообще везде good movie             |

**UPD**: T=100, beta = 0.1 то результаты будут лучше в генерации: (2.4282062350586058, 1.5289127522706984) (kl с reward упадут но будут более разнообразные генерации)

**UPD**: T=500, beta = 0.1 (2.7279955463111403, 17.49258479118347) (Новое предложение всегда начинается с I was so excited to see this movie, но завершения промтов всегда окей)

**UPD**: T=1000, beta = 0.1 (2.6261013615876436, 6.324007024765015) (циклятся(( )


### RLOO
betas = 0.01
k = 10
| T   | reward      | kl          | нормальность генераций на мой глаз  |
|-----|-------------|-------------|-------------------------------------|
| 100 | 2.41        | 3.18        | окей, генерит нормальные предложения|
| 500 | 1.96        | 23.38       | нормик                              |
| 500, beta=0.1|  1.9 |  124.56           |      хорошие сэмплы, правда не понятно куда kl улетела) |


## How to run
sad, но я не успел написать скрипты для тренировки sft модельки со всеми экспериментами поэтому:

**Все ноутбуки можно запустить на kaggle (только после reward-train, нужно сохранить reward модельку и всё)**

**run reward-train.ipynb**

**run warp.ipynb**

**run warp-rloo.ipynb**

---
Потренировать reward модельку на парах (взял по 1000 с единичек и нулей получился лям сэмплов (кстати скорее всего это не оч правильно потому что в реальности человеческая разметка в том же instruct gpt была около 1000))
```
sh run.sh
```
Дальше, ноутбук с WARP (и три эксперимента, которые запускаются если реверсать ноутбук с разными параметрами (sad), но я для разных экспериментов сделал отдельные ячейки)
run warp.ipynb
Дальше, ноутбук с WARP + RLOO
run warp-rloo.ipynb


## Hardware
немного печально что такие резы с такой карточкой, но да)
- GPU: 1x NVIDIA A100 (81920MiB)
