# T-Lab-2024-LLM-Alignment

## Mini-batch итогов
- Сначала я реализовал метод reinforce (sorry trl в вашем коде я нереально запутался)
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

## Таблица

### WARP

| T  | kl          | reward      | нормальность генераций на мой глаз |
|----|-------------|-------------|-------------------------------------|
| 100 | Значение 1  | Значение 2  | окей                               |
| 500 | Значение 4  | Значение 5  | Значение 6                         |
| 1000 | Значение 7  | Значение 8  | FUCCCCK                           |

### RLOO

| T  | kl          | reward      | нормальность генераций на мой глаз |
|----|-------------|-------------|-------------------------------------|
| 100 | Значение 1  | Значение 2  | Значение 3                         |
| 500 | Значение 4  | Значение 5  | Значение 6                         |
| 1000 | Значение 7  | Значение 8  | Значение 9                        |


## How to run
sad, но я не успел написать скрипты для тренировки sft модельки со всеми экспериментами поэтому:

Потренировать reward модельку на парах (взял по 1000 с единичек и нулей получился лям сэмплов (кстати скорее всего это не оч правильно потому что в реальности человеческая разметка в том же instruct gpt была около 1000))
```
sh run.sh
```
Дальше, ноутбук с WARP (и три эксперимента)
run warp.ipynb
Дальше, ноутбук с WARP + RLOO
run rloo.ipynb



## Hardware
немного печально что такие резы с такой карточкой, но да)
Nvidia A100 80gb
