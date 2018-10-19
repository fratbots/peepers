from state import STATE_BEGINNING, STATE_LEVEL_ONE, STATE_LEVEL_TWO, STATE_GAME_OVER
from game import Game

game = Game(
    eye_ar_thresh=0.2,
    eye_ar_consec_frames=3
)

while True:
    if game.state() == STATE_BEGINNING:
        game.beginning()
    if game.state() == STATE_LEVEL_ONE:
        game.level_one()
    if game.state() == STATE_LEVEL_TWO:
        game.level_two()
    if game.state() == STATE_GAME_OVER:
        game.over()
        break
