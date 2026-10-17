"""Phrase-aware interruption: ignore lists, break-in lists and the greeting's threshold."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
)
from pipecat.turns.empty_user_turn import (
    DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT,
    EmptyUserTurnConfig,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.phrase_turn_start import (
    PhraseAwareUserTurnStartStrategy,
    PhraseList,
    create_turn_start_strategy,
    normalize_phrases,
    normalize_transcript,
    resolve_turn_start_phrases,
)
from api.services.pipecat.run_pipeline import (
    RESUME_EMPTY_TURN_PROMPT,
    _create_non_realtime_user_turn_start_strategies,
    _resolve_empty_user_turn,
    _resolve_greeting_min_words,
)

IGNORE = [
    "uh-huh",
    "mm-hmm",
    "mhm",
    "yeah",
    "yep",
    "okay",
    "ok",
    "right",
    "sure",
    "i see",
    "got it",
]
BREAK_IN = [
    "stop",
    "wait",
    "hold on",
    "hello",
    "hang on",
    "excuse me",
    "sorry",
    "no",
    "actually",
    "one second",
]


# --- normalisation ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Uh-huh.", "uh huh"),
        ("Mm-hmm, yeah!", "mm hmm yeah"),
        ("  Hold   ON?? ", "hold on"),
        ("I'm here, actually.", "im here actually"),
        ("", ""),
    ],
)
def test_normalize_transcript_lowercases_and_strips_punctuation_and_hyphens(
    text, expected
):
    assert normalize_transcript(text) == expected


def test_normalize_phrases_drops_blanks_and_duplicates_in_order():
    assert normalize_phrases(["Uh-huh", " ", "uh huh", "Yeah!", None, "yeah"]) == [
        "uh huh",
        "yeah",
    ]


def test_phrase_list_covers_only_a_run_of_listed_phrases():
    phrases = PhraseList(IGNORE)
    assert phrases.covers(["yeah"])
    assert phrases.covers(["yeah", "yeah"])
    assert phrases.covers(["okay", "i", "see"])
    assert phrases.covers(["uh", "huh"])
    assert not phrases.covers(["yeah", "but"])
    assert not phrases.covers(["i"])
    assert not phrases.covers([])
    assert not PhraseList([]).covers(["yeah"])


def test_phrase_list_appears_in_matches_a_contiguous_run():
    phrases = PhraseList(BREAK_IN)
    assert phrases.appears_in(["well", "actually", "i", "wanted"])
    assert phrases.appears_in(["please", "hold", "on"])
    assert not phrases.appears_in(["hold", "the", "line"])
    assert not phrases.appears_in([])


# --- the strategy ------------------------------------------------------------


class Observed:
    def __init__(self, strategy):
        self.starts = 0
        self.resets = 0

        @strategy.event_handler("on_user_turn_started")
        async def started(_strategy, _params):
            self.starts += 1

        @strategy.event_handler("on_reset_aggregation")
        async def reset(_strategy):
            self.resets += 1


def strategy_with(min_words=3, ignore=IGNORE, break_in=BREAK_IN):
    strategy = PhraseAwareUserTurnStartStrategy(
        min_words=min_words, ignore_phrases=ignore, break_in_phrases=break_in
    )
    return strategy, Observed(strategy)


def transcript(text, interim=False):
    cls = InterimTranscriptionFrame if interim else TranscriptionFrame
    return cls(text=text, user_id="caller", timestamp="")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Uh-huh.", "yeah yeah", "Okay, I see.", "MM-HMM"])
async def test_ignore_only_transcript_never_starts_a_turn_while_bot_speaks(text):
    strategy, seen = strategy_with(min_words=1)
    await strategy.process_frame(BotStartedSpeakingFrame())
    result = await strategy.process_frame(transcript(text))
    assert result is ProcessFrameResult.CONTINUE
    assert seen.starts == 0
    assert seen.resets == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Stop.", "Hold on!", "well, actually", "Hello"])
async def test_break_in_phrase_starts_the_turn_below_the_word_count(text):
    strategy, seen = strategy_with(min_words=5)
    await strategy.process_frame(BotStartedSpeakingFrame())
    result = await strategy.process_frame(transcript(text))
    assert result is ProcessFrameResult.STOP
    assert seen.starts == 1
    assert seen.resets == 0


@pytest.mark.asyncio
async def test_mixed_transcript_with_a_break_in_word_interrupts_even_with_back_channels():
    strategy, seen = strategy_with(min_words=5)
    await strategy.process_frame(BotStartedSpeakingFrame())
    assert (
        await strategy.process_frame(transcript("yeah yeah wait"))
        is ProcessFrameResult.STOP
    )
    assert seen.starts == 1


@pytest.mark.asyncio
async def test_unlisted_words_fall_back_to_the_word_count():
    strategy, seen = strategy_with(min_words=3)
    await strategy.process_frame(BotStartedSpeakingFrame())
    assert (
        await strategy.process_frame(transcript("yeah but"))
        is ProcessFrameResult.CONTINUE
    )
    assert seen.starts == 0
    assert seen.resets == 1
    assert (
        await strategy.process_frame(transcript("yeah but the price"))
        is ProcessFrameResult.STOP
    )
    assert seen.starts == 1


@pytest.mark.asyncio
async def test_interim_transcripts_are_judged_the_same_way():
    strategy, seen = strategy_with(min_words=3)
    await strategy.process_frame(BotStartedSpeakingFrame())
    assert (
        await strategy.process_frame(transcript("uh-huh", interim=True))
        is ProcessFrameResult.CONTINUE
    )
    assert (
        await strategy.process_frame(transcript("hold on", interim=True))
        is ProcessFrameResult.STOP
    )
    assert seen.starts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Uh-huh.", "Stop.", "one"])
async def test_when_the_bot_is_not_speaking_one_word_opens_a_turn_as_before(text):
    strategy, seen = strategy_with(min_words=3)
    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(BotStoppedSpeakingFrame())
    assert await strategy.process_frame(transcript(text)) is ProcessFrameResult.STOP
    assert seen.starts == 1
    assert seen.resets == 0


@pytest.mark.asyncio
async def test_an_empty_transcript_is_reset_as_before():
    strategy, seen = strategy_with(min_words=1)
    await strategy.process_frame(BotStartedSpeakingFrame())
    assert await strategy.process_frame(transcript("")) is ProcessFrameResult.CONTINUE
    assert seen.starts == 0
    assert seen.resets == 1


# --- wiring ------------------------------------------------------------------


def test_create_turn_start_strategy_returns_the_plain_class_without_phrases():
    strategy = create_turn_start_strategy(
        min_words=2, ignore_phrases=[" "], break_in_phrases=None
    )
    assert type(strategy) is MinWordsUserTurnStartStrategy
    assert strategy._min_words == 2


def test_create_turn_start_strategy_returns_the_phrase_aware_class_with_a_list():
    strategy = create_turn_start_strategy(min_words=3, ignore_phrases=["Uh-huh"])
    assert isinstance(strategy, PhraseAwareUserTurnStartStrategy)
    assert strategy._ignore.phrases == ["uh huh"]
    assert not strategy._break_in


@pytest.mark.parametrize(
    "configs,expected",
    [
        ({}, ([], [])),
        (
            {"turn_start_ignore_phrases": None, "turn_start_break_in_phrases": "stop"},
            ([], []),
        ),
        (
            {
                "turn_start_ignore_phrases": ["Yeah", ""],
                "turn_start_break_in_phrases": ["Hold on"],
            },
            (["yeah"], ["hold on"]),
        ),
    ],
)
def test_resolve_turn_start_phrases(configs, expected):
    assert resolve_turn_start_phrases(configs) == expected


@pytest.mark.parametrize("uses_external_turns", [False, True])
def test_min_words_strategy_becomes_phrase_aware_when_a_list_is_set(
    uses_external_turns,
):
    strategies = _create_non_realtime_user_turn_start_strategies(
        {
            "turn_start_strategy": "min_words",
            "turn_start_min_words": 3,
            "turn_start_break_in_phrases": ["stop"],
        },
        uses_external_turns=uses_external_turns,
    )
    assert len(strategies) == 1
    assert isinstance(strategies[0], PhraseAwareUserTurnStartStrategy)
    assert strategies[0]._min_words == 3
    assert strategies[0]._break_in.phrases == ["stop"]


def test_empty_phrase_lists_keep_the_plain_min_words_strategy():
    strategies = _create_non_realtime_user_turn_start_strategies(
        {"turn_start_ignore_phrases": [], "turn_start_break_in_phrases": []},
        uses_external_turns=False,
    )
    assert type(strategies[0]) is MinWordsUserTurnStartStrategy


@pytest.mark.parametrize(
    "configs,expected",
    [
        ({}, 2),
        ({"greeting_min_words": None}, 2),
        ({"greeting_min_words": 1}, 1),
        ({"greeting_min_words": 0}, 1),
    ],
)
def test_resolve_greeting_min_words(configs, expected):
    assert _resolve_greeting_min_words(configs) == expected


def test_empty_turn_recovery_default_is_pipecats_own():
    assert _resolve_empty_user_turn({}) == EmptyUserTurnConfig()
    assert (
        _resolve_empty_user_turn({"empty_turn_recovery": "repeat"})
        == LLMUserAggregatorParams().empty_user_turn
    )
    assert (
        _resolve_empty_user_turn({}).interrupted_prompt
        == DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT
    )


def test_empty_turn_recovery_resume_and_silent():
    resumed = _resolve_empty_user_turn({"empty_turn_recovery": "resume"})
    assert resumed.interrupted_prompt == RESUME_EMPTY_TURN_PROMPT
    assert resumed.idle_prompt is None
    assert resumed.max_consecutive_recoveries == 1
    assert _resolve_empty_user_turn({"empty_turn_recovery": "silent"}) is None
    assert (
        _resolve_empty_user_turn({"empty_turn_recovery": "bogus"})
        == EmptyUserTurnConfig()
    )


# --- the greeting controller --------------------------------------------------


class FakeController:
    def __init__(self):
        self.user_turn_strategies = UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=9)],
            stop=[ExternalUserTurnStopStrategy()],
        )
        self.has_active_user_turn = False
        self.installed = []

    async def update_strategies(self, strategies):
        self.user_turn_strategies = strategies
        self.installed.append(strategies)


def greeting_controller(*, greeting_pending, screening=False):
    playback = SimpleNamespace(greeting_pending=greeting_pending, greeting=None)
    controller = GreetingController(
        playback, lambda: None, is_screening=lambda: screening
    )
    user = SimpleNamespace(
        user_turn_controller=FakeController(),
        add_event_handler=lambda *_: None,
        create_task=lambda *_: None,
        queue_frame=AsyncMock(),
    )
    controller.bind(user)
    return controller, user, playback


@pytest.mark.asyncio
async def test_greeting_installs_the_two_word_threshold_when_not_configured():
    controller, user, _ = greeting_controller(greeting_pending=True)
    await controller._update_strategies(user, Frame())
    [installed] = user.user_turn_controller.installed
    [start] = installed.start
    assert type(start) is MinWordsUserTurnStartStrategy
    assert start._min_words == 2
    assert start._enable_interruptions is True
    assert start._bot_speaking is True


@pytest.mark.asyncio
async def test_greeting_uses_the_configured_threshold_and_phrase_lists():
    controller, user, _ = greeting_controller(greeting_pending=True)
    controller.configure(
        min_words=1, ignore_phrases=["Uh-huh"], break_in_phrases=["stop"]
    )
    await controller._update_strategies(user, Frame())
    [start] = user.user_turn_controller.user_turn_strategies.start
    assert isinstance(start, PhraseAwareUserTurnStartStrategy)
    assert start._min_words == 1
    assert start._ignore.phrases == ["uh huh"]
    assert start._break_in.phrases == ["stop"]
    assert start._enable_interruptions is True


@pytest.mark.asyncio
async def test_screening_keeps_its_single_word_policy_without_phrases():
    controller, user, _ = greeting_controller(greeting_pending=False, screening=True)
    controller.configure(
        min_words=1, ignore_phrases=["uh-huh"], break_in_phrases=["stop"]
    )
    await controller._update_strategies(user, Frame())
    [start] = user.user_turn_controller.user_turn_strategies.start
    assert type(start) is MinWordsUserTurnStartStrategy
    assert start._min_words == 1
    assert start._enable_interruptions is False


@pytest.mark.asyncio
async def test_a_one_word_greeting_threshold_still_replaces_the_screening_policy():
    # Both policies want one word; keying on the threshold alone would leave
    # the screening strategy, which never interrupts, in place for the greeting.
    screening = True
    controller, user, playback = greeting_controller(
        greeting_pending=False, screening=True
    )
    controller._is_screening = lambda: screening
    controller.configure(min_words=1)
    await controller._update_strategies(user, Frame())
    assert (
        user.user_turn_controller.user_turn_strategies.start[0]._enable_interruptions
        is False
    )
    screening = False
    playback.greeting_pending = True
    await controller._update_strategies(user, Frame())
    [start] = user.user_turn_controller.user_turn_strategies.start
    assert start._enable_interruptions is True
    assert start._min_words == 1
    assert len(user.user_turn_controller.installed) == 2


@pytest.mark.asyncio
async def test_normal_strategies_come_back_after_the_greeting():
    controller, user, playback = greeting_controller(greeting_pending=True)
    normal = user.user_turn_controller.user_turn_strategies
    await controller._update_strategies(user, Frame())
    assert user.user_turn_controller.user_turn_strategies is not normal
    playback.greeting_pending = False
    await controller._update_strategies(user, Frame())
    assert user.user_turn_controller.user_turn_strategies is normal
