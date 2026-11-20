"""After the greeting, the normal turn rules know the agent has stopped.

Copy runs 6999 and 7004 (2026-11-20): the recorded-line notice started the
agent speaking while the normal turn-start rules were installed; the greeting
then swapped its own rules in and the notice's stop went to those. Put back
after the greeting, the normal rules still believed the agent was speaking,
so a "Hi." or a "Yeah." after the greeting was dropped and never answered.
"""

from types import SimpleNamespace

import pytest

from pipecat.frames.frames import BotStartedSpeakingFrame, Frame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.phrase_turn_start import create_turn_start_strategy


@pytest.mark.asyncio
async def test_a_single_word_after_the_greeting_opens_a_turn():
    start = create_turn_start_strategy(
        min_words=3, ignore_phrases=["yeah", "okay"], break_in_phrases=["stop"], enable_interruptions=True
    )
    normal = UserTurnStrategies(start=[start], stop=[SpeechTimeoutUserTurnStopStrategy()])
    installed: list[UserTurnStrategies] = []

    async def update_strategies(strategies):
        installed.append(strategies)

    user = SimpleNamespace(
        user_turn_controller=SimpleNamespace(
            user_turn_strategies=normal, update_strategies=update_strategies, has_active_user_turn=False
        ),
        add_event_handler=lambda *_: None,
    )
    playback = SimpleNamespace(greeting_pending=False, greeting=None)
    greeting = GreetingController(playback, lambda: None, is_screening=lambda: False)
    greeting.bind(user)

    # The notice plays under the normal rules.
    await start.process_frame(BotStartedSpeakingFrame())
    # The greeting is queued, plays and ends.
    playback.greeting_pending = True
    await greeting._update_strategies(user, Frame())
    playback.greeting_pending = False
    await greeting._update_strategies(user, Frame())
    assert installed[-1] is normal

    for word in ("Hi.", "Yeah."):
        result = await start.process_frame(TranscriptionFrame(word, "caller", "2026-11-20T23:25:40Z"))
        assert result == ProcessFrameResult.STOP, f"{word!r} after the greeting was dropped"
